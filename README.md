# Diarized Transcriber

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/diarized-transcriber.svg)](https://pypi.org/project/diarized-transcriber/)

A Python library for transcribing audio and video with automatic speaker diarization. Uses [WhisperX](https://github.com/m-bain/whisperX) for high-quality speech-to-text and [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker detection, labeling, and fingerprinting.

## Key Features

- **Speech-to-text transcription** using WhisperX (faster-whisper / CTranslate2)
- **Speaker diarization** — automatically detect and label who said what
- **Speaker fingerprinting** — enroll speakers and recognize them across recordings using voice embeddings
- **Cross-recording speaker identification** — match speakers between different audio files
- **REST API** — built-in FastAPI server for transcription and speaker management
- **GPU accelerated** — PyTorch + CUDA for fast inference
- **Multiple output formats** — text and dictionary/JSON
- **Configurable model sizes** — tiny, base, small, medium, large

## How It Works

1. Audio is loaded and preprocessed
2. WhisperX transcribes speech to text with word-level timestamps
3. pyannote.audio detects speaker turns (diarization)
4. Speaker embeddings are extracted using WeSpeaker ResNet34-LM (256-dim)
5. Embeddings are matched against enrolled speaker profiles in LanceDB for identification
6. Results are returned with speaker labels, timestamps, and optional speaker names

## Installation

```bash
pip install diarized-transcriber
```

With optional extras:

```bash
# FastAPI server
pip install diarized-transcriber[server]

# Speaker profiles (fingerprinting + cross-recording ID)
pip install diarized-transcriber[profiles]

# Everything
pip install diarized-transcriber[server,profiles]
```

## Requirements

- Python 3.10+
- CUDA-capable GPU
- PyTorch with CUDA support
- HuggingFace token for pyannote.audio model access

```bash
export HF_TOKEN="<your-huggingface-token>"
```

## Quick Start

```python
from diarized_transcriber import TranscriptionEngine, MediaContent, MediaSource

engine = TranscriptionEngine(model_size="base")

content = MediaContent(
    id="example-1",
    title="Example Media",
    media_url="https://example.com/media.mp3",
    source=MediaSource(type="podcast")
)

result = engine.transcribe(content)

from diarized_transcriber.utils.formatting import format_transcript
print(format_transcript(result, output_format="text", group_by_speaker=True))
```

## Configuration

The `TranscriptionEngine` accepts different Whisper model sizes:

| Model    | Speed   | Accuracy |
|----------|---------|----------|
| `tiny`   | Fastest | Lowest   |
| `base`   | Fast    | Good     |
| `small`  | Medium  | Better   |
| `medium` | Slow    | High     |
| `large`  | Slowest | Highest  |

```python
engine = TranscriptionEngine(
    model_size="medium",
    compute_type="float16"  # or "float32" for higher precision
)
```

## FastAPI Server

```bash
pip install diarized-transcriber[server]
python -m diarized_transcriber.api.server
```

The server runs on port 8000 and provides endpoints for transcription and speaker management (`/speakers/*`).

## Speaker Management API

When installed with the `profiles` extra, the API exposes endpoints for managing speaker profiles:

- `POST /speakers/enroll` — Enroll a new speaker from audio
- `GET /speakers/` — List all enrolled speakers
- `GET /speakers/{id}` — Get speaker profile details
- `PUT /speakers/{id}` — Update a speaker profile
- `DELETE /speakers/{id}` — Remove a speaker profile
- `POST /speakers/identify` — Identify a speaker from audio
- `POST /speakers/merge` — Merge two speaker profiles

## Output Formats

### Text

```python
transcript = format_transcript(
    result,
    output_format="text",
    include_timestamps=True,
    timestamp_format="HH:MM:SS.mmm",
    group_by_speaker=True
)
```

### Dictionary

```python
transcript_dict = format_transcript(
    result,
    output_format="dict",
    include_timestamps=True
)
```

## Error Handling

The library provides specific exceptions:

- `GPUConfigError` — GPU/CUDA configuration issues
- `ModelLoadError` — Model loading failures
- `AudioProcessingError` — Audio processing problems
- `TranscriptionError` — General transcription failures
- `DiarizationError` — Speaker diarization issues

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License — see [LICENSE](LICENSE) for details.
