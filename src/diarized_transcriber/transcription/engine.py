"""Main transcription engine coordinating WhisperX and speaker diarization."""

import logging
from pathlib import Path
from typing import Optional, Literal
import torch
import whisperx
import pandas as pd
from ..exceptions import (
    ModelLoadError,
    TranscriptionError,
    GPUConfigError
)
from ..models.content import MediaContent
from ..models.transcription import (
    TranscriptionResult,
    TranscriptionSegment,
    Speaker
)
from .gpu import verify_gpu_requirements, cleanup_gpu_memory
from .audio import load_audio, process_media_content, create_temp_audio_file
from .diarization import DiarizationPipeline
from .embeddings import SpeakerEmbeddingExtractor
from ..identification.matcher import SpeakerIdentifier
from ..storage.speaker_store import SpeakerProfileStore
from ..types import AudioArray

logger = logging.getLogger(__name__)

class TranscriptionEngine:
    """Coordinates transcription and diarization of media content."""

    def __init__(
        self,
        model_size: Literal["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "large-v3-turbo"] = "large-v3-turbo",
        device: Optional[str] = None,
        compute_type: Literal["float16", "float32"] = "float16",
        extract_embeddings: bool = True,
        speaker_db_path: Optional[str] = None,
        identify_speakers: bool = False,
        auto_enroll_speakers: bool = False,
        match_threshold: float = 0.4
    ):
        """
        Initialize the transcription engine.

        Args:
            model_size: WhisperX model size to use
            device: Computing device (defaults to CUDA if available)
            compute_type: Model computation type
            extract_embeddings: Whether to extract speaker embeddings
            speaker_db_path: Path to speaker profile database directory
            identify_speakers: Whether to identify speakers against stored profiles
            auto_enroll_speakers: Whether to auto-enroll unmatched speakers
            match_threshold: Cosine distance threshold for speaker matching (0.0-2.0)

        Raises:
            GPUConfigError: If GPU verification fails
            ValueError: If identify_speakers=True but speaker_db_path is None
        """
        if identify_speakers and speaker_db_path is None:
            raise ValueError(
                "speaker_db_path is required when identify_speakers=True"
            )
        if match_threshold < 0.0 or match_threshold > 2.0:
            raise ValueError(
                f"match_threshold must be between 0.0 and 2.0, got {match_threshold}"
            )
        if auto_enroll_speakers and not identify_speakers:
            logger.warning(
                "auto_enroll_speakers=True has no effect when "
                "identify_speakers=False"
            )

        self.model_size = model_size
        self.compute_type = compute_type
        self.extract_embeddings = extract_embeddings
        self.speaker_db_path = speaker_db_path
        self.identify_speakers = identify_speakers
        self.auto_enroll_speakers = auto_enroll_speakers
        self.match_threshold = match_threshold
        self.device = device or verify_gpu_requirements()

        # Initialize components as needed
        self._whisper_model = None
        self._align_model = None
        self._diarization = None
        self._embedding_extractor = None
        self._speaker_identifier = None

    def _initialize_whisper(self) -> None:
        """
        Initialize WhisperX model if not already initialized.

        Raises:
            ModelLoadError: If model initialization fails
        """
        if self._whisper_model is None:
            try:
                logger.info("Loading WhisperX model: %s", self.model_size)
                self._whisper_model = whisperx.load_model(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type
                )
            except Exception as e:
                raise ModelLoadError(
                    f"Failed to load WhisperX model: {str(e)}"
                ) from e

    def _initialize_diarization(self) -> None:
        """Initialize diarization pipeline if not already initialized."""
        if self._diarization is None:
            self._diarization = DiarizationPipeline(device=self.device)

    def _run_pipeline(
        self,
        audio_array: AudioArray,
        sample_rate: int,
        content_id: str,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> TranscriptionResult:
        """
        Run the core transcription + diarization pipeline on loaded audio.

        Args:
            audio_array: Loaded audio samples
            sample_rate: Sample rate of the audio
            content_id: Identifier for the content being transcribed
            language: Language code hint (unused currently, detected by whisper)
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect

        Returns:
            TranscriptionResult with segments, speakers, and metadata
        """
        # Initial transcription
        logger.info("Starting transcription")
        result = self._whisper_model.transcribe(audio_array)

        # Align whisper output
        logger.info("Aligning transcription")
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=self.device
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio_array,
            self.device,
            return_char_alignments=False
        )

        # Create temporary file for diarization
        temp_path, temp_file = create_temp_audio_file(
            audio_array,
            sample_rate
        )

        try:
            # Perform diarization
            logger.info("Performing speaker diarization")
            speakers = self._diarization.process_audio(
                temp_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )

            # Convert segments to DataFrame for speaker assignment
            segments_df = pd.DataFrame([
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"]
                }
                for segment in result["segments"]
            ])

            # Assign speakers to segments
            segments_df = self._diarization.assign_speakers_to_segments(
                speakers,
                segments_df
            )

            # Extract speaker embeddings
            if self.extract_embeddings:
                logger.info("Extracting speaker embeddings")
                if self._embedding_extractor is None:
                    self._embedding_extractor = SpeakerEmbeddingExtractor(
                        device=self.device
                    )
                self._embedding_extractor.extract_for_all_speakers(
                    temp_path, speakers
                )

        finally:
            # Clean up temporary file
            import os
            os.unlink(temp_path)

        # Identify speakers against stored profiles
        if self.identify_speakers and self.speaker_db_path:
            logger.info("Identifying speakers against stored profiles")
            if self._speaker_identifier is None:
                store = SpeakerProfileStore(self.speaker_db_path)
                self._speaker_identifier = SpeakerIdentifier(
                    store=store,
                    match_threshold=self.match_threshold,
                    auto_enroll=self.auto_enroll_speakers,
                )
            id_results = self._speaker_identifier.identify_speakers(speakers)
            for speaker, profile in id_results:
                if profile is not None and profile.name:
                    speaker.metadata["original_speaker_id"] = speaker.id
                    speaker.id = profile.name

        # Create final TranscriptionResult
        transcription_segments = [
            TranscriptionSegment(
                start=row["start"],
                end=row["end"],
                text=row["text"],
                speaker=row["speaker"]
            )
            for _, row in segments_df.iterrows()
        ]

        return TranscriptionResult(
            content_id=content_id,
            language=result["language"],
            segments=transcription_segments,
            speakers=speakers,
            metadata={
                "model_size": self.model_size,
                "compute_type": self.compute_type
            }
        )

    def transcribe(
        self,
        content: MediaContent,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> TranscriptionResult:
        """
        Transcribe media content with speaker diarization.

        Args:
            content: MediaContent to transcribe
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect

        Returns:
            TranscriptionResult: Complete transcription with speaker information

        Raises:
            TranscriptionError: If transcription process fails
        """
        try:
            # Initialize components
            self._initialize_whisper()
            self._initialize_diarization()

            # Load and process audio
            logger.info("Processing audio for: %s", content.title)
            audio_array, sample_rate = process_media_content(content)

            final_result = self._run_pipeline(
                audio_array,
                sample_rate,
                content_id=content.id,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

            # Cleanup
            cleanup_gpu_memory()

            return final_result

        except Exception as e:
            cleanup_gpu_memory()
            raise TranscriptionError(
                f"Transcription failed: {str(e)}"
            ) from e

    def transcribe_file(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file with speaker diarization.

        Args:
            audio_path: Path to the audio file
            language: Language code hint
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect

        Returns:
            TranscriptionResult: Complete transcription with speaker information

        Raises:
            TranscriptionError: If transcription process fails
        """
        try:
            self._initialize_whisper()
            self._initialize_diarization()

            logger.info("Processing audio file: %s", audio_path)
            audio_array, sample_rate = load_audio(str(audio_path))

            final_result = self._run_pipeline(
                audio_array,
                sample_rate,
                content_id=str(audio_path),
                language=language,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

            cleanup_gpu_memory()
            return final_result

        except Exception as e:
            cleanup_gpu_memory()
            raise TranscriptionError(
                f"Transcription failed: {str(e)}"
            ) from e
