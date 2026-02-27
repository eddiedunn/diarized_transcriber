"""Custom exceptions for the diarized_transcriber package."""

class DiarizedTranscriberError(Exception):
    """Base exception for all diarized_transcriber errors."""
    pass

class GPUConfigError(DiarizedTranscriberError):
    """Raised when there are issues with GPU configuration."""
    pass

class ModelLoadError(DiarizedTranscriberError):
    """Raised when there are issues loading ML models."""
    pass

class AudioProcessingError(DiarizedTranscriberError):
    """Raised when there are issues processing audio."""
    pass

class TranscriptionError(DiarizedTranscriberError):
    """Raised when there are issues during transcription."""
    pass

class DiarizationError(DiarizedTranscriberError):
    """Raised when there are issues during speaker diarization."""
    pass

class EmbeddingExtractionError(DiarizedTranscriberError):
    """Raised when there are issues extracting speaker embeddings."""
    pass

class StorageError(DiarizedTranscriberError):
    """Raised when there are issues with speaker profile storage."""
    pass
