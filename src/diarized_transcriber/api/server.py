"""HTTP server exposing transcription capabilities via FastAPI."""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, Form

from ..models.content import MediaContent, MediaSource

from .schemas import TranscriptionRequest, TranscriptionResponse

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Send our (and gpu_common's) log records to stderr, i.e. the container log.

    Without this the root logger has no handler: INFO is dropped and the
    tracebacks from logger.exception() never reach the journal.
    """
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

# Try to import the GPU-backed backend; it is optional so tests and
# lightweight installations can still import this module. The backend runs all GPU
# work in a worker process; this process must never touch CUDA.
_backend = None
_backend_error: Optional[str] = None
try:
    from .backend import GPUOutOfMemory, get_backend

    _backend = get_backend()
except Exception as exc:  # reported at startup, once logging is configured
    _backend_error = repr(exc)

    class GPUOutOfMemory(RuntimeError):  # type: ignore[no-redef]
        pass


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Manage backend startup and shutdown."""
    _configure_logging()
    if _backend is None:
        logger.error("GPU backend unavailable: %s", _backend_error)
    if _backend is not None:
        await _backend.start()
        logger.info("Backend started")
    yield
    if _backend is not None:
        await _backend.stop()
        logger.info("Backend stopped")


app = FastAPI(title="Diarized Transcriber", lifespan=lifespan)

# Include speaker management routes if available
try:
    from .speaker_routes import router as speaker_router
    app.include_router(speaker_router)
except (ImportError, AttributeError):
    pass

# Include fingerprint routes if available
try:
    from .fingerprint_routes import router as fingerprint_router
    app.include_router(fingerprint_router)
except (ImportError, AttributeError):
    pass


@app.get("/")
def read_root() -> dict[str, str]:
    """Basic health endpoint."""

    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(req: TranscriptionRequest) -> TranscriptionResponse:
    """Transcribe the provided media file (legacy; runs in the GPU worker too)."""

    if _backend is None:
        raise HTTPException(status_code=503, detail="Backend not available")
    try:
        content = MediaContent(
            id=req.id,
            title=req.title,
            media_url=req.media_url,
            source=MediaSource(type="local"),
        )

        result = await _backend.transcribe_content(
            content,
            req.model_size or "large-v3-turbo",
            min_speakers=req.min_speakers,
            max_speakers=req.max_speakers,
        )
        return TranscriptionResponse(result=result)
    except Exception as exc:  # pragma: no cover - fastapi handles
        logger.exception("transcribe failed")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


# ── v1 endpoints ──────────────────────────────────────────────────────


@app.post("/v1/transcribe")
async def v1_transcribe(
    audio: UploadFile,
    language: Optional[str] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    cleanup: bool = Form(False),
    include_embeddings: bool = Form(False),
    identify_speakers: Optional[bool] = Form(None),
    auto_enroll_speakers: Optional[bool] = Form(None),
) -> dict:
    """Transcribe an uploaded audio file (multipart form).

    Compatible with curator's orchestrator contract:
    ``files={"audio": ...}, data={"cleanup": "true"}``
    """
    if _backend is None:
        raise HTTPException(
            status_code=503,
            detail="Backend not available (gpu-common not installed)",
        )

    tmp_path = None
    try:
        suffix = os.path.splitext(audio.filename or "upload.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as tmp:
            tmp_path = tmp.name
            contents = await audio.read()
            tmp.write(contents)

        from pathlib import Path

        result = await _backend.transcribe(
            Path(tmp_path),
            language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            identify_speakers=identify_speakers,
            auto_enroll_speakers=auto_enroll_speakers,
        )

        # Build flat response shape expected by curator
        segments = []
        for seg in result.segments:
            segments.append(
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "speaker": seg.speaker,
                }
            )

        speakers = []
        if result.speakers:
            for spk in result.speakers:
                segment_count = len(
                    [s for s in result.segments if s.speaker == spk.id]
                )
                entry = {
                    "id": spk.id,
                    "name": spk.metadata.get("original_speaker_id"),
                    "profile_id": spk.metadata.get("profile_id"),
                    "segment_count": segment_count,
                }
                if include_embeddings and spk.embedding is not None:
                    entry["embedding"] = spk.embedding
                speakers.append(entry)

        # Compute total duration from segments
        duration = 0.0
        if result.segments:
            duration = max(s.end for s in result.segments)

        return {
            "text": " ".join(s.text for s in result.segments),
            "segments": segments,
            "language": result.language,
            "duration": duration,
            "model": result.metadata.get("model_size", "unknown"),
            "speakers": speakers,
        }
    except TimeoutError as exc:
        logger.warning("GPU lock timeout — service busy")
        raise HTTPException(
            status_code=503,
            detail="Transcription service busy, please retry later",
            headers={"Retry-After": "60"},
        ) from exc
    except GPUOutOfMemory as exc:
        logger.error("v1/transcribe: GPU out of memory after retry: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="GPU out of memory, please retry later",
            headers={"Retry-After": "120"},
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("v1/transcribe failed")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/health")
async def health() -> dict:
    """Service health check. Must not touch CUDA (see backend)."""
    model_loaded = False
    gpu_available = False
    if _backend is not None:
        model_loaded = _backend.worker.is_loaded
        try:
            from gpu_common import check_gpu_available

            gpu_available = check_gpu_available()
        except Exception:
            pass
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "gpu_available": gpu_available,
    }


@app.get("/models")
async def models() -> dict:
    """List available model sizes."""
    return {
        "models": [
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v2",
            "large-v3",
            "large-v3-turbo",
        ]
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    try:
        from ..config import get_settings

        settings = get_settings()
        host = settings.host
        port = settings.port
    except Exception:
        host = "0.0.0.0"
        port = 8000

    uvicorn.run(
        "diarized_transcriber.api.server:app",
        host=host,
        port=port,
        reload=False,
    )
