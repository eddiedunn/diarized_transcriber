# Diarized Transcriber

A Python library for transcribing media content with speaker diarization support. This package provides high-quality transcription using WhisperX with automatic speaker detection and labeling.

## Features

- High-quality transcription using WhisperX
- Automatic speaker diarization
- Support for multiple media sources
- GPU acceleration support
- Flexible output formats
- Type-safe interfaces

## Installation

```bash
pip install diarized-transcriber
```

## Requirements

- Python 3.10 or later
- CUDA-capable GPU
- PyTorch with CUDA support
- HuggingFace account for `pyannote.audio` access

Before running the examples below you must export an authentication token for
the diarization pipeline. This token is referenced in
`diarized_transcriber/transcription/diarization.py` (see lines 43–55) and is
required for downloading the pretrained `pyannote.audio` model.

```bash
export HF_TOKEN="<your-huggingface-token>"
```

## Quick Start

```python
from diarized_transcriber import TranscriptionEngine, MediaContent, MediaSource

# Initialize the engine
engine = TranscriptionEngine(model_size="base")

# Create media content object
content = MediaContent(
    id="example-1",
    title="Example Media",
    media_url="https://example.com/media.mp3",
    source=MediaSource(type="podcast")
)

# Perform transcription
result = engine.transcribe(content)

# Format the result
from diarized_transcriber.utils.formatting import format_transcript
transcript = format_transcript(
    result,
    output_format="text",
    group_by_speaker=True
)

print(transcript)
```

## Environment Setup

Set up your HuggingFace token for `pyannote.audio` access:

```bash
export HF_TOKEN='your-huggingface-token'
```

Ensure CUDA is properly configured for your system. A good way to check if CUDA is working is to run `nvidia-smi` in your terminal.

## Configuration

The `TranscriptionEngine` can be configured with different model sizes:

- `tiny`: Fastest, lowest accuracy
- `base`: Good balance of speed and accuracy
- `small`: Better accuracy, slower than base
- `medium`: High accuracy, slower
- `large`: Highest accuracy, slowest

Example configuration:

```python
engine = TranscriptionEngine(
    model_size="medium",
    compute_type="float16"  # or "float32" for higher precision
)
```

## Output Formats

The library supports multiple output formats.

### Text Format

```python
transcript = format_transcript(
    result,
    output_format="text",
    include_timestamps=True,
    timestamp_format="HH:MM:SS.mmm",
    group_by_speaker=True
)
```

### Dictionary Format

```python
transcript_dict = format_transcript(
    result,
    output_format="dict",
    include_timestamps=True
)
```

## Running the FastAPI server

Install the optional dependencies:

```bash
pip install diarized-transcriber[server]
```

Start the service:

```bash
python -m diarized_transcriber.api.server
```

## Error Handling

The library provides specific exceptions for different error cases:

- `GPUConfigError`: GPU-related issues
- `ModelLoadError`: Model loading failures
- `AudioProcessingError`: Audio processing problems
- `TranscriptionError`: General transcription failures
- `DiarizationError`: Speaker diarization issues

## Contributing

Contributions are welcome! Please see our contributing guidelines for more details.

## License

MIT License - see `LICENSE` file for details.
