"""HTTP server exposing transcription capabilities via FastAPI."""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, Form

from ..models.content import MediaContent, MediaSource
from ..transcription.engine import TranscriptionEngine

from .schemas import TranscriptionRequest, TranscriptionResponse

logger = logging.getLogger(__name__)

# Try to import the GPU-backed backend; it is optional so tests and
# lightweight installations can still import this module.
_backend = None
try:
    from .backend import DiarizedTranscriberBackend

    _backend = DiarizedTranscriberBackend()
except Exception:
    pass


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Manage backend startup and shutdown."""
    if _backend is not None:
        await _backend._manager.start()
        await _backend._lock.connect()
        logger.info("Backend started")
    yield
    if _backend is not None:
        await _backend._manager.stop()
        await _backend._lock.disconnect()
        logger.info("Backend stopped")


app = FastAPI(title="Diarized Transcriber", lifespan=lifespan)

# Include speaker management routes if available
try:
    from .speaker_routes import router as speaker_router
    app.include_router(speaker_router)
except (ImportError, AttributeError):
    pass


@app.get("/")
def read_root() -> dict[str, str]:
    """Basic health endpoint."""

    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscriptionResponse)
def transcribe(req: TranscriptionRequest) -> TranscriptionResponse:
    """Transcribe the provided media file."""

    try:
        engine = TranscriptionEngine(
            model_size=req.model_size or "large-v3-turbo"
        )

        content = MediaContent(
            id=req.id,
            title=req.title,
            media_url=req.media_url,
            source=MediaSource(type="local"),
        )

        result = engine.transcribe(
            content,
            min_speakers=req.min_speakers,
            max_speakers=req.max_speakers,
        )
        return TranscriptionResponse(result=result)
    except Exception as exc:  # pragma: no cover - fastapi handles
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── v1 endpoints ──────────────────────────────────────────────────────


@app.post("/v1/transcribe")
async def v1_transcribe(
    audio: UploadFile,
    language: Optional[str] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    cleanup: bool = Form(False),
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
                speakers.append(
                    {
                        "id": spk.id,
                        "name": spk.metadata.get("original_speaker_id"),
                        "segment_count": segment_count,
                    }
                )

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
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("v1/transcribe failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/health")
async def health() -> dict:
    """Service health check."""
    model_loaded = False
    gpu_available = False
    if _backend is not None:
        model_loaded = _backend._manager._model is not None
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
