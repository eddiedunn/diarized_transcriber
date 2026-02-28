"""Pydantic models for the speaker management API endpoints."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class EnrollSpeakerRequest(BaseModel):
    """Request body for enrolling a new speaker profile."""

    embedding: List[float] = Field(
        description="Speaker embedding vector (256-dim)"
    )
    name: Optional[str] = Field(default=None, max_length=200)

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v):
        if len(v) != 256:
            raise ValueError(
                f"Embedding must be 256-dimensional, got {len(v)}"
            )
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Embedding values must be numeric")
        return v


class NameSpeakerRequest(BaseModel):
    """Request body for renaming a speaker profile."""

    name: str = Field(min_length=1, max_length=200)

    @field_validator("name")
    @classmethod
    def strip_name(cls, v):
        return v.strip()


class MergeProfilesRequest(BaseModel):
    """Request body for merging two speaker profiles."""

    source_profile_id: str = Field(min_length=1)
    target_profile_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def source_not_target(self):
        if self.source_profile_id == self.target_profile_id:
            raise ValueError(
                "source_profile_id and target_profile_id must be different"
            )
        return self


class SearchSpeakersRequest(BaseModel):
    """Request body for searching speaker profiles by embedding similarity."""

    embedding: List[float] = Field(
        description="Query embedding (256-dim)"
    )
    limit: int = Field(default=10, ge=1, le=50)
    threshold: float = Field(default=0.4, ge=0.0, le=2.0)

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v):
        if len(v) != 256:
            raise ValueError(
                f"Embedding must be 256-dimensional, got {len(v)}"
            )
        return v


class SpeakerProfileResponse(BaseModel):
    """Response model for a single speaker profile."""

    id: str
    name: Optional[str]
    embedding: List[float]
    enrollment_count: int
    total_duration: Optional[float]
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SpeakerProfileListResponse(BaseModel):
    """Response model for a list of speaker profiles."""

    profiles: List[SpeakerProfileResponse]
    count: int


class SpeakerSearchResponse(BaseModel):
    """Response model for speaker search results."""

    matches: List[dict]  # Each has profile_id, distance, confidence, profile
    count: int
