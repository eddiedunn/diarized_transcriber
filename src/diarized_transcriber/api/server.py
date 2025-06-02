"""HTTP server exposing transcription capabilities via FastAPI."""

from fastapi import FastAPI, HTTPException

from ..models.content import MediaContent, MediaSource
from ..transcription.engine import TranscriptionEngine

from .schemas import TranscriptionRequest, TranscriptionResponse


app = FastAPI(title="Diarized Transcriber")


@app.get("/")
def read_root() -> dict[str, str]:
    """Basic health endpoint."""

    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscriptionResponse)
def transcribe(req: TranscriptionRequest) -> TranscriptionResponse:
    """Transcribe the provided media file."""

    try:
        engine = TranscriptionEngine(
            model_size=req.model_size or "base"
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


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "diarized_transcriber.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )

