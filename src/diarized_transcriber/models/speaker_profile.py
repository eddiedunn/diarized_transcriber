"""Speaker profile models for cross-recording speaker identification."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .transcription import EMBEDDING_DIM


class SpeakerProfile(BaseModel):
    """Persistent speaker profile with embedding for cross-recording identification."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = Field(default=None, max_length=200)
    embedding: List[float] = Field(
        description="Speaker embedding vector (256-dim, L2-normalized)"
    )
    enrollment_count: int = Field(default=1, ge=1)
    total_duration: Optional[float] = Field(default=None, ge=0.0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def strip_name(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
        return v

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: List[float]) -> List[float]:
        if len(v) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding must be {EMBEDDING_DIM}-dimensional, got {len(v)}"
            )
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Embedding values must be numeric (int or float)")
        return v


class SpeakerMatch(BaseModel):
    """Result of matching a speaker embedding against stored profiles."""

    profile_id: str = Field(min_length=1)
    distance: float = Field(ge=0.0, le=2.0)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def compute_confidence(cls, v: float, info) -> float:
        if v is not None:
            return v
        distance = info.data.get("distance", 0.0)
        return 1.0 - distance
