"""Utilities for formatting transcription results."""

import logging
from typing import Union, Optional
from datetime import timedelta
from ..models.transcription import TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)

def format_timestamp(seconds: float, include_ms: bool = False) -> str:
    """
    Format time in seconds to HH:MM:SS or HH:MM:SS.mmm.
    
    Args:
        seconds: Time in seconds
        include_ms: Whether to include milliseconds
        
    Returns:
        str: Formatted timestamp
    """
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    milliseconds = td.microseconds // 1000
    
    if include_ms:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def format_segment(
    segment: TranscriptionSegment,
    include_timestamps: bool = True,
    timestamp_format: str = "HH:MM:SS"
) -> str:
    """
    Format a single transcription segment.
    
    Args:
        segment: Segment to format
        include_timestamps: Whether to include timing information
        timestamp_format: Format for timestamps ("HH:MM:SS" or "HH:MM:SS.mmm")
        
    Returns:
        str: Formatted segment text
    """
    text = segment.text.strip()
    if not include_timestamps:
        return f"[{segment.speaker or 'UNKNOWN'}] {text}"
    
    include_ms = timestamp_format == "HH:MM:SS.mmm"
    start = format_timestamp(segment.start, include_ms)
    end = format_timestamp(segment.end, include_ms)
    
    return f"[{start} -> {end}] [{segment.speaker or 'UNKNOWN'}] {text}"

def format_transcript(
    result: TranscriptionResult,
    output_format: str = "text",
    include_timestamps: bool = True,
    timestamp_format: str = "HH:MM:SS",
    group_by_speaker: bool = True
) -> Union[str, dict]:
    """
    Format complete transcription result.
    
    Args:
        result: TranscriptionResult to format
        output_format: Format type ("text" or "dict")
        include_timestamps: Whether to include timing information
        timestamp_format: Format for timestamps ("HH:MM:SS" or "HH:MM:SS.mmm")
        group_by_speaker: Whether to group segments by speaker
        
    Returns:
        Union[str, dict]: Formatted transcript
    """
    if output_format == "dict":
        return _format_transcript_dict(
            result,
            include_timestamps,
            timestamp_format
        )
    
    return _format_transcript_text(
        result,
        include_timestamps,
        timestamp_format,
        group_by_speaker
    )

def _format_transcript_dict(
    result: TranscriptionResult,
    include_timestamps: bool,
    timestamp_format: str
) -> dict:
    """Format transcript as a dictionary."""
    return {
        "content_id": result.content_id,
        "language": result.language,
        "segments": [
            {
                "start": format_timestamp(
                    s.start,
                    timestamp_format == "HH:MM:SS.mmm"
                ),
                "end": format_timestamp(
                    s.end,
                    timestamp_format == "HH:MM:SS.mmm"
                ),
                "speaker": s.speaker or "UNKNOWN",
                "text": s.text.strip()
            }
            for s in result.segments
        ],
        "metadata": result.metadata
    }

def _format_transcript_text(
    result: TranscriptionResult,
    include_timestamps: bool,
    timestamp_format: str,
    group_by_speaker: bool
) -> str:
    """Format transcript as formatted text."""
    if not group_by_speaker:
        return "\n".join(
            format_segment(s, include_timestamps, timestamp_format)
            for s in result.segments
        )
    
    # Group segments by speaker
    speaker_segments: dict[str, list[str]] = {}
    for segment in result.segments:
        speaker = segment.speaker or "UNKNOWN"
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        
        speaker_segments[speaker].append(
            format_segment(
                segment,
                include_timestamps,
                timestamp_format
            )
        )
    
    # Build final output
    output = []
    for speaker, segments in speaker_segments.items():
        output.append(f"\nSpeaker {speaker}:")
        output.extend(segments)
    
    return "\n".join(output)
