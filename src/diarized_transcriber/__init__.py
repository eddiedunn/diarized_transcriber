"""
diarized_transcriber - A library for transcribing media content with speaker diarization
"""

import logging
from importlib.metadata import version, PackageNotFoundError
from dotenv import load_dotenv

load_dotenv()

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

# Set up package-level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Import key types for user convenience
from .models.content import MediaContent, MediaSource
from .models.transcription import (
    TranscriptionResult,
    TranscriptionSegment,
    Speaker
)
from .exceptions import (
    DiarizedTranscriberError,
    GPUConfigError,
    AudioProcessingError,
    TranscriptionError,
    DiarizationError
)
