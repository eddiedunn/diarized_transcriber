"""GPU-backed transcription backend.

All GPU work runs in a ``GPUWorker`` child process (see ``gpu_handler``) that is
stopped after ``idle_timeout``: ending the process is the only way to be sure the
GPU memory of whisperx, pyannote and CTranslate2 goes back to the driver. This web
process must never touch CUDA itself.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from gpu_common import GPULock, GPUOutOfMemoryError, GPUWorker, is_gpu_oom  # noqa: F401

from ..config import get_settings
from ..models.content import MediaContent
from ..models.transcription import Speaker, TranscriptionResult

logger = logging.getLogger(__name__)

HANDLER = "diarized_transcriber.api.gpu_handler:TranscriberHandler"


class GPUOutOfMemory(RuntimeError):
    """The GPU ran out of memory even in a fresh worker process."""


class DiarizedTranscriberBackend:
    """Runs transcriptions in the GPU worker, behind the shared GPU lock."""

    def __init__(self) -> None:
        settings = get_settings()
        self._lock = GPULock(settings.redis_url)
        self.worker = GPUWorker(
            HANDLER, idle_timeout=settings.idle_timeout, name="transcription worker"
        )

    async def start(self) -> None:
        await self.worker.start()
        await self._lock.connect()

    async def stop(self) -> None:
        await self.worker.stop()
        await self._lock.disconnect()

    async def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        identify_speakers: Optional[bool] = None,
        auto_enroll_speakers: Optional[bool] = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file, holding the GPU lock.

        A GPU out-of-memory error stops the worker; retry once in a fresh one and raise
        GPUOutOfMemory if that runs out too.
        """
        kwargs = dict(
            language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            identify_speakers_override=identify_speakers,
            auto_enroll_override=auto_enroll_speakers,
        )
        async with self._lock.acquire(timeout=1800, blocking_timeout=600):
            return await self._with_oom_retry("transcribe_file", str(audio_path), **kwargs)

    async def transcribe_content(
        self,
        content: MediaContent,
        model_size: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> TranscriptionResult:
        async with self._lock.acquire(timeout=1800, blocking_timeout=600):
            return await self._with_oom_retry(
                "transcribe_content", content, model_size,
                min_speakers=min_speakers, max_speakers=max_speakers,
            )

    async def fingerprint(
        self, audio_path: Path, speakers: list[Speaker]
    ) -> list[tuple[Optional[list[float]], Optional[str]]]:
        """Short job: no GPU lock, like before; it waits behind a running transcription."""
        return await self._with_oom_retry("fingerprint", str(audio_path), speakers)

    async def _with_oom_retry(self, method: str, *args, **kwargs):
        try:
            return await self.worker.call(method, *args, **kwargs)
        except GPUOutOfMemoryError as exc:
            logger.warning("GPU out of memory in %s (%s); retrying once in a fresh worker", method, exc)
        try:
            return await self.worker.call(method, *args, **kwargs)
        except GPUOutOfMemoryError as exc:
            raise GPUOutOfMemory(str(exc)) from exc


@lru_cache
def get_backend() -> DiarizedTranscriberBackend:
    return DiarizedTranscriberBackend()
