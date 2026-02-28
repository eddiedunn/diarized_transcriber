import sys
import types
import importlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

# Stub minimal pydantic so modules can be imported without dependency
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
pydantic.HttpUrl = str
sys.modules.setdefault("pydantic", pydantic)

# Provide a lightweight package stub for diarized_transcriber
pkg = types.ModuleType("diarized_transcriber")
# Point the package path at the real source directory so submodules load
pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "diarized_transcriber")]
sys.modules.setdefault("diarized_transcriber", pkg)

# Import the module under test
formatting = importlib.import_module("diarized_transcriber.utils.formatting")

@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

@dataclass
class Result:
    content_id: str
    language: str
    segments: List[Segment]
    metadata: dict = field(default_factory=dict)


def test_format_timestamp():
    assert formatting.format_timestamp(3661.234) == "01:01:01"
    assert formatting.format_timestamp(3661.234, include_ms=True) == "01:01:01.234"


def test_format_segment():
    seg_with_speaker = Segment(0, 1.5, "Hello", "A")
    assert (
        formatting.format_segment(seg_with_speaker)
        == "[00:00:00 -> 00:00:01] [A] Hello"
    )

    seg_no_speaker = Segment(0, 2, "Hi there", None)
    assert formatting.format_segment(seg_no_speaker, include_timestamps=False) == "[UNKNOWN] Hi there"


def test_format_transcript_dict_and_text():
    seg1 = Segment(0, 1, " Hello ", "S1")
    seg2 = Segment(1.5, 3, "World", None)
    result = Result("c1", "en", [seg1, seg2], {"foo": "bar"})

    d = formatting._format_transcript_dict(result, True, "HH:MM:SS")
    assert d == {
        "content_id": "c1",
        "language": "en",
        "segments": [
            {"start": "00:00:00", "end": "00:00:01", "speaker": "S1", "text": "Hello"},
            {"start": "00:00:01", "end": "00:00:03", "speaker": "UNKNOWN", "text": "World"},
        ],
        "metadata": {"foo": "bar"},
    }

    text = formatting._format_transcript_text(result, True, "HH:MM:SS", True)
    expected = (
        "\nSpeaker S1:\n"
        "[00:00:00 -> 00:00:01] [S1] Hello\n\n"
        "Speaker UNKNOWN:\n"
        "[00:00:01 -> 00:00:03] [UNKNOWN] World"
    )
    assert text == expected
