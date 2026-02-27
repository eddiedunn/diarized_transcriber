import sys
import types
import importlib
from pathlib import Path
from datetime import timedelta

# Stub minimal pydantic like in utils tests
pydantic = types.ModuleType("pydantic")
class BaseModel:
    def __init__(self, **data):
        for k, v in data.items():
            setattr(self, k, v)

def Field(default=None, *, default_factory=None, **_kwargs):
    if default_factory is not None:
        return default_factory()
    return default
def field_validator(*args, **kwargs):
    def decorator(fn):
        return classmethod(fn) if not isinstance(fn, classmethod) else fn
    return decorator
pydantic.BaseModel = BaseModel
pydantic.Field = Field
pydantic.field_validator = field_validator
sys.modules.setdefault("pydantic", pydantic)

# Provide lightweight package stub for diarized_transcriber
pkg = types.ModuleType("diarized_transcriber")
pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "diarized_transcriber")]
sys.modules.setdefault("diarized_transcriber", pkg)

# Import module under test
transcription = importlib.import_module("diarized_transcriber.models.transcription")
TimeSegment = transcription.TimeSegment
Speaker = transcription.Speaker
EMBEDDING_DIM = transcription.EMBEDDING_DIM


def test_time_segment_to_timedelta():
    seg = TimeSegment(start=1.5, end=3.0)
    start_td, end_td = seg.to_timedelta()
    assert start_td == timedelta(seconds=1.5)
    assert end_td == timedelta(seconds=3.0)


def test_speaker_embedding_field_default_none():
    speaker = Speaker(id="A", segments=[])
    assert speaker.embedding is None


def test_speaker_embedding_accepts_valid_256_dim():
    emb = [0.1] * 256
    speaker = Speaker(id="A", segments=[], embedding=emb)
    assert speaker.embedding == emb
    assert len(speaker.embedding) == 256


def test_speaker_embedding_rejects_wrong_dimension():
    import pytest
    with pytest.raises(ValueError):
        Speaker.validate_embedding([0.1] * 128)


def test_speaker_embedding_rejects_non_numeric():
    """validate_embedding rejects non-numeric element types even at correct dimension."""
    import pytest
    with pytest.raises(ValueError):
        Speaker.validate_embedding(["a"] * 256)


def test_speaker_serialization_with_embedding():
    emb = [0.5] * 256
    speaker = Speaker(id="B", segments=[], embedding=emb)
    assert speaker.embedding is not None
    assert speaker.embedding == emb
    assert len(speaker.embedding) == EMBEDDING_DIM


def test_speaker_serialization_without_embedding():
    speaker = Speaker(id="C", segments=[])
    assert speaker.embedding is None
