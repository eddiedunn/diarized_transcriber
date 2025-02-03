from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import timedelta

class TimeSegment(BaseModel):
    """Represents a time segment in the transcription"""
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    
    def to_timedelta(self) -> tuple[timedelta, timedelta]:
        """Convert start and end times to timedelta objects"""
        return (timedelta(seconds=self.start), timedelta(seconds=self.end))

class TranscriptionSegment(TimeSegment):
    """Represents a segment of transcribed text with timing"""
    text: str = Field(description="Transcribed text for this segment")
    speaker: Optional[str] = Field(
        default=None,
        description="Speaker identifier for this segment"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score for this transcription"
    )

class Speaker(BaseModel):
    """Represents a speaker identified in the content"""
    id: str = Field(description="Unique identifier for the speaker")
    segments: List[TimeSegment] = Field(
        description="Time segments where this speaker is active"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Speaker-specific metadata"
    )

class TranscriptionResult(BaseModel):
    """Complete transcription result for a piece of media content"""
    content_id: str = Field(description="ID of the transcribed content")
    language: str = Field(description="Detected language code")
    segments: List[TranscriptionSegment] = Field(
        description="Sequential transcription segments"
    )
    speakers: Optional[List[Speaker]] = Field(
        default=None,
        description="Identified speakers and their segments"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional transcription metadata"
    )