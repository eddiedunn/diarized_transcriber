"""Speaker diarization functionality using pyannote.audio."""

import logging
import os
from typing import Optional
import torch
from pyannote.audio import Pipeline
import pandas as pd
from ..exceptions import DiarizationError
from ..models.transcription import Speaker, TimeSegment

logger = logging.getLogger(__name__)

class DiarizationPipeline:
    """Manages the speaker diarization pipeline."""
    
    def __init__(
        self,
        device: str = "cuda",
        model: str = "pyannote/speaker-diarization-3.1"
    ):
        """
        Initialize the diarization pipeline.
        
        Args:
            device: Computing device ('cuda' or 'cpu')
            model: Model identifier for pyannote.audio
            
        Raises:
            DiarizationError: If pipeline initialization fails
        """
        self.device = device
        self.model = model
        self._pipeline: Optional[Pipeline] = None
        
    def _initialize_pipeline(self) -> None:
        """
        Initialize the pyannote pipeline if not already initialized.
        
        Raises:
            DiarizationError: If initialization fails
        """
        if self._pipeline is None:
            try:
                auth_token = os.environ.get("HF_TOKEN")
                if not auth_token:
                    raise DiarizationError(
                        "HF_TOKEN environment variable not set. "
                        "Required for pyannote.audio access."
                    )
                
                self._pipeline = Pipeline.from_pretrained(
                    self.model,
                    token=auth_token
                )
                self._pipeline = self._pipeline.to(torch.device(self.device))
                
            except Exception as e:
                raise DiarizationError(
                    f"Failed to initialize diarization pipeline: {str(e)}"
                ) from e
    
    def process_audio(
        self,
        audio_file: Optional[str] = None,
        audio_array: Optional["AudioArray"] = None,
        sample_rate: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> list[Speaker]:
        """
        Process audio and identify speakers.

        Accepts either a file path or pre-loaded audio data. When both are
        provided, audio_array takes precedence (avoids pyannote 4.x
        torchcodec/AudioDecoder dependency).

        Args:
            audio_file: Path to audio file
            audio_array: Pre-loaded audio samples (numpy array)
            sample_rate: Sample rate of audio_array (required with audio_array)
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect

        Returns:
            list[Speaker]: List of identified speakers with their segments

        Raises:
            DiarizationError: If diarization fails
        """
        try:
            self._initialize_pipeline()

            # Build pipeline input: prefer waveform dict (pyannote 4.x compat)
            if audio_array is not None:
                if sample_rate is None:
                    raise DiarizationError(
                        "sample_rate is required when passing audio_array"
                    )
                import numpy as np
                waveform = torch.from_numpy(np.asarray(audio_array)).float()
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                audio_input = {"waveform": waveform, "sample_rate": sample_rate}
            elif audio_file is not None:
                audio_input = audio_file
            else:
                raise DiarizationError(
                    "Either audio_file or audio_array must be provided"
                )

            logger.info("Starting speaker diarization")
            diarize_output = self._pipeline(
                audio_input,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )

            # pyannote 4.x returns DiarizeOutput dataclass;
            # 3.x returned Annotation directly. Handle both.
            if hasattr(diarize_output, "speaker_diarization"):
                annotation = diarize_output.speaker_diarization
            else:
                annotation = diarize_output

            # Convert pyannote output to our data model
            speakers: dict[str, list[TimeSegment]] = {}

            for segment, _, speaker in annotation.itertracks(yield_label=True):
                if speaker not in speakers:
                    speakers[speaker] = []
                    
                speakers[speaker].append(
                    TimeSegment(
                        start=segment.start,
                        end=segment.end
                    )
                )
            
            return [
                Speaker(
                    id=speaker_id,
                    segments=segments
                )
                for speaker_id, segments in speakers.items()
            ]
            
        except Exception as e:
            raise DiarizationError(
                f"Speaker diarization failed: {str(e)}"
            ) from e
    
    def assign_speakers_to_segments(
        self,
        speakers: list[Speaker],
        segments: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Assign speaker labels to transcription segments.
        
        Args:
            speakers: List of identified speakers
            segments: DataFrame with start/end times and transcription
            
        Returns:
            pd.DataFrame: Segments with speaker labels assigned
            
        Raises:
            DiarizationError: If speaker assignment fails
        """
        try:
            # Create a mapping of time ranges to speakers
            speaker_ranges = []
            for speaker in speakers:
                for segment in speaker.segments:
                    speaker_ranges.append({
                        'start': segment.start,
                        'end': segment.end,
                        'speaker': speaker.id
                    })
            
            speaker_df = pd.DataFrame(speaker_ranges)
            
            # Function to find speaker for a given segment
            def find_speaker(row):
                matches = speaker_df[
                    (speaker_df['start'] <= row['start']) &
                    (speaker_df['end'] >= row['end'])
                ]
                return matches['speaker'].iloc[0] if not matches.empty else None
            
            # Add speaker column to segments
            segments['speaker'] = segments.apply(find_speaker, axis=1)
            
            return segments
            
        except Exception as e:
            raise DiarizationError(
                f"Failed to assign speakers to segments: {str(e)}"
            ) from e
