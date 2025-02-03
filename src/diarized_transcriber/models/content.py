from typing import Dict, Optional, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl

class MediaSource(BaseModel):
    """Represents the origin of media content"""
    type: Literal["podcast", "youtube", "local"] = Field(
        description="Type of media source"
    )
    url: Optional[HttpUrl] = Field(
        default=None,
        description="Source URL (e.g. RSS feed URL, YouTube channel URL)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific metadata"
    )

class MediaContent(BaseModel):
    """Represents a piece of media content to be transcribed"""
    id: str = Field(description="Unique identifier for the content")
    title: str = Field(description="Title of the content")
    media_url: HttpUrl = Field(description="Direct URL to the media file")
    duration: Optional[float] = Field(
        default=None,
        description="Duration in seconds"
    )
    source: MediaSource = Field(
        description="Information about the content source"
    )
    published_date: Optional[datetime] = Field(
        default=None,
        description="When the content was published"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Content-specific metadata"
    )