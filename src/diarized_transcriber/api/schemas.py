"""Pydantic models for the API layer."""

from typing import Optional, Literal
from pydantic import BaseModel, Field, HttpUrl

from ..models.transcription import TranscriptionResult


class TranscriptionRequest(BaseModel):
    """Request body for the transcription endpoint."""

    id: str = Field(description="Unique identifier for the content")
    title: str = Field(description="Title of the content")
    media_url: HttpUrl = Field(description="URL to the media file")
    model_size: Optional[
        Literal["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "large-v3-turbo"]
    ] = Field(
        default=None,
        description="WhisperX model size to use"
    )
    min_speakers: Optional[int] = Field(
        default=None, description="Minimum number of speakers"
    )
    max_speakers: Optional[int] = Field(
        default=None, description="Maximum number of speakers"
    )


class TranscriptionResponse(BaseModel):
    """Response returned by the transcription endpoint."""

    result: TranscriptionResult = Field(
        description="Transcription result for the requested media"
    )

