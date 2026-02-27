"""Speaker embedding extraction using pyannote.audio."""

import logging
import os
from typing import Optional

import numpy as np
from pyannote.audio import Inference, Model
from pyannote.core import Segment

from ..exceptions import DiarizationError, EmbeddingExtractionError
from ..models.transcription import EMBEDDING_DIM, Speaker

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
MIN_SEGMENT_DURATION = 0.5  # seconds
MIN_TOTAL_SPEECH = 1.0  # seconds


class SpeakerEmbeddingExtractor:
    """Extracts speaker embeddings from audio segments using pyannote."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model: Optional[Model] = None
        self._inference: Optional[Inference] = None

    def _initialize(self) -> None:
        """Lazy-initialize the embedding model."""
        if self._model is not None:
            return

        auth_token = os.environ.get("HF_TOKEN")
        if not auth_token:
            raise DiarizationError(
                "HF_TOKEN environment variable not set. "
                "Required for pyannote.audio access."
            )

        self._model = Model.from_pretrained(
            EMBEDDING_MODEL, token=auth_token
        )
        self._inference = Inference(self._model, window="whole")
        self._inference.to(self.device)

    def extract_for_speaker(
        self, audio_file: str, speaker: Speaker
    ) -> list[float]:
        """
        Extract a single embedding for a speaker by averaging segment embeddings.

        Args:
            audio_file: Path to the audio file.
            speaker: Speaker with time segments.

        Returns:
            L2-normalized 256-dim embedding as a list of floats.

        Raises:
            EmbeddingExtractionError: If no valid segments or insufficient speech.
        """
        self._initialize()

        if not speaker.segments:
            raise EmbeddingExtractionError(
                f"Speaker {speaker.id} has no segments"
            )

        # Filter segments by minimum duration
        valid_segments = []
        for seg in speaker.segments:
            duration = seg.end - seg.start
            if duration < MIN_SEGMENT_DURATION:
                logger.warning(
                    "Skipping short segment (%.2fs) for speaker %s",
                    duration,
                    speaker.id,
                )
                continue
            valid_segments.append(seg)

        # Check total speech duration
        total_duration = sum(s.end - s.start for s in valid_segments)
        if total_duration < MIN_TOTAL_SPEECH:
            raise EmbeddingExtractionError(
                f"Speaker {speaker.id} has insufficient speech "
                f"({total_duration:.2f}s < {MIN_TOTAL_SPEECH}s)"
            )

        # Extract embeddings from each valid segment
        embeddings = []
        for seg in valid_segments:
            try:
                segment = Segment(seg.start, seg.end)
                emb = self._inference.crop(audio_file, segment)
                embeddings.append(np.array(emb).flatten())
            except Exception as e:
                logger.warning(
                    "Failed to extract embedding for speaker %s segment "
                    "[%.2f-%.2f]: %s",
                    speaker.id,
                    seg.start,
                    seg.end,
                    str(e),
                )

        if not embeddings:
            raise EmbeddingExtractionError(
                f"All segment extractions failed for speaker {speaker.id}"
            )

        # Average and L2-normalize
        mean_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm

        result = mean_embedding.tolist()
        if len(result) != EMBEDDING_DIM:
            raise EmbeddingExtractionError(
                f"Expected {EMBEDDING_DIM}-dim embedding, got {len(result)}"
            )
        return result

    def extract_for_all_speakers(
        self, audio_file: str, speakers: list[Speaker]
    ) -> list[Speaker]:
        """
        Extract embeddings for all speakers, updating their embedding fields.

        Graceful degradation: if extraction fails for a speaker, logs a warning
        and leaves embedding as None.

        Args:
            audio_file: Path to the audio file.
            speakers: List of Speaker objects to update.

        Returns:
            The same list of Speaker objects with embeddings populated where possible.
        """
        for speaker in speakers:
            try:
                embedding = self.extract_for_speaker(audio_file, speaker)
                speaker.embedding = embedding
            except Exception as e:
                logger.warning(
                    "Could not extract embedding for speaker %s: %s",
                    speaker.id,
                    str(e),
                )
        return speakers
