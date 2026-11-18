"""Everything that touches the GPU, run inside gpu_common's worker process.

The web server never loads models itself: it calls these methods through a
``GPUWorker``, which stops the process when idle so the driver gets all of its GPU
memory back. Arguments and return values cross a process boundary and must be
picklable.
"""

import logging
from pathlib import Path
from typing import Optional

from ..config import get_settings
from ..models.content import MediaContent
from ..models.transcription import Speaker, TranscriptionResult

logger = logging.getLogger(__name__)


class TranscriberHandler:
    """Loads each model on first use; the worker process lifetime bounds them."""

    def __init__(self) -> None:
        self._engine = None
        self._extractor = None

    def _get_engine(self):
        if self._engine is None:
            from ..transcription.engine import TranscriptionEngine

            settings = get_settings()
            self._engine = TranscriptionEngine(
                model_size=settings.model_size,
                extract_embeddings=settings.extract_embeddings,
                speaker_db_path=settings.speaker_db_path,
                identify_speakers=settings.identify_speakers,
                auto_enroll_speakers=settings.auto_enroll_speakers,
                match_threshold=settings.match_threshold,
            )
        return self._engine

    def transcribe_file(self, audio_path: str, **kwargs) -> TranscriptionResult:
        return self._get_engine().transcribe_file(Path(audio_path), **kwargs)

    def transcribe_content(
        self,
        content: MediaContent,
        model_size: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> TranscriptionResult:
        """Legacy /transcribe: a URL instead of an upload, optionally another model size."""
        if model_size == get_settings().model_size:
            engine = self._get_engine()
        else:
            from ..transcription.engine import TranscriptionEngine

            engine = TranscriptionEngine(model_size=model_size)
        return engine.transcribe(content, min_speakers=min_speakers, max_speakers=max_speakers)

    def fingerprint(
        self, audio_path: str, speakers: list[Speaker]
    ) -> list[tuple[Optional[list[float]], Optional[str]]]:
        """One (embedding, error) pair per speaker; an error never fails the batch."""
        if self._extractor is None:
            from ..transcription.embeddings import SpeakerEmbeddingExtractor

            self._extractor = SpeakerEmbeddingExtractor(device="cuda")
        results: list[tuple[Optional[list[float]], Optional[str]]] = []
        for speaker in speakers:
            try:
                results.append((self._extractor.extract_for_speaker(audio_path, speaker), None))
            except Exception as exc:
                logger.warning("Embedding extraction failed for %s: %s", speaker.id, exc)
                results.append((None, str(exc)))
        return results
