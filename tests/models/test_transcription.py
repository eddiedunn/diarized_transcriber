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
pydantic.BaseModel = BaseModel
pydantic.Field = Field
sys.modules.setdefault("pydantic", pydantic)

# Provide lightweight package stub for diarized_transcriber
pkg = types.ModuleType("diarized_transcriber")
pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "diarized_transcriber")]
sys.modules.setdefault("diarized_transcriber", pkg)

# Import module under test
transcription = importlib.import_module("diarized_transcriber.models.transcription")
TimeSegment = transcription.TimeSegment


def test_time_segment_to_timedelta():
    seg = TimeSegment(start=1.5, end=3.0)
    start_td, end_td = seg.to_timedelta()
    assert start_td == timedelta(seconds=1.5)
    assert end_td == timedelta(seconds=3.0)
