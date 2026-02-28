"""GPU-backed transcription backend with resource management."""

import asyncio
import logging
from datetime import timedelta
from pathlib import Path
from typing import Optional

from gpu_common import GPULock, GPUModelManager

from ..config import get_settings
from ..models.transcription import TranscriptionResult
from ..transcription.engine import TranscriptionEngine

logger = logging.getLogger(__name__)


class DiarizedTranscriberBackend:
    """Backend that manages a TranscriptionEngine behind a GPU lock."""

    def __init__(self) -> None:
        settings = get_settings()
        self._lock = GPULock(settings.redis_url)
        self._manager: GPUModelManager[TranscriptionEngine] = GPUModelManager(
            load_fn=self._load_engine,
            idle_timeout=timedelta(seconds=settings.idle_timeout),
        )

    def _load_engine(self) -> TranscriptionEngine:
        settings = get_settings()
        return TranscriptionEngine(
            model_size=settings.model_size,
            extract_embeddings=settings.extract_embeddings,
            speaker_db_path=settings.speaker_db_path,
            identify_speakers=settings.identify_speakers,
            auto_enroll_speakers=settings.auto_enroll_speakers,
            match_threshold=settings.match_threshold,
        )

    async def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file, acquiring the GPU lock first."""
        async with self._lock.acquire(timeout=7200, blocking_timeout=600):
            engine = await self._manager.get()
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: engine.transcribe_file(
                    audio_path,
                    language=language,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                ),
            )
        return result
