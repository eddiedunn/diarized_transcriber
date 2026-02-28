import sys
import types
import importlib
from pathlib import Path

# Stub minimal pydantic like other tests
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

# Stub torch and pyannote.audio so imports succeed
torch = types.ModuleType("torch")
def device(x):
    return x
torch.device = device
sys.modules.setdefault("torch", torch)

pyannote = types.ModuleType("pyannote")
pyannote_audio = types.ModuleType("pyannote.audio")

class DummyPipeline:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def to(self, _device):
        return self

class DummyModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

class DummyInference:
    def __init__(self, model, window="whole"):
        self.model = model
    def to(self, _device):
        return self
    def crop(self, audio_file, segment):
        return [0.0] * 256

pyannote_audio.Pipeline = DummyPipeline
pyannote_audio.Model = DummyModel
pyannote_audio.Inference = DummyInference
pyannote.audio = pyannote_audio

pyannote_core = types.ModuleType("pyannote.core")
class DummySegment:
    def __init__(self, start, end):
        self.start = start
        self.end = end
pyannote_core.Segment = DummySegment
pyannote.core = pyannote_core

sys.modules.setdefault("pyannote", pyannote)
sys.modules.setdefault("pyannote.audio", pyannote_audio)
sys.modules.setdefault("pyannote.core", pyannote_core)

# Minimal pandas stub used by diarization.assign_speakers_to_segments
pandas = types.ModuleType("pandas")

class Series:
    def __init__(self, data):
        self.data = list(data)

    def __le__(self, other):
        return Series([x <= other for x in self.data])

    def __ge__(self, other):
        return Series([x >= other for x in self.data])

    def __and__(self, other):
        return Series([a and b for a, b in zip(self.data, other.data)])

    @property
    def iloc(self):
        class ILoc:
            def __init__(self, data):
                self._data = data
            def __getitem__(self, idx):
                return self._data[idx]
        return ILoc(self.data)

    @property
    def empty(self):
        return len(self.data) == 0

    def __iter__(self):
        return iter(self.data)

class DataFrame:
    def __init__(self, data):
        self.data = [dict(row) for row in data]

    def __getitem__(self, key):
        if isinstance(key, str):
            return Series([row.get(key) for row in self.data])
        if isinstance(key, Series):
            filtered = [row for row, flag in zip(self.data, key.data) if flag]
            return DataFrame(filtered)
        raise TypeError("Unsupported key type")

    def __setitem__(self, key, value):
        values = value.data if isinstance(value, Series) else value
        for row, val in zip(self.data, values):
            row[key] = val

    def apply(self, func, axis=0):
        assert axis == 1
        return Series([func(row) for row in self.data])

    @property
    def empty(self):
        return len(self.data) == 0

pandas.DataFrame = DataFrame
sys.modules.setdefault("pandas", pandas)

# Provide a lightweight package stub for diarized_transcriber
pkg = types.ModuleType("diarized_transcriber")
pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "diarized_transcriber")]
sys.modules.setdefault("diarized_transcriber", pkg)

# Import modules under test
models = importlib.import_module("diarized_transcriber.models.transcription")
Speaker = models.Speaker
TimeSegment = models.TimeSegment
pipeline_mod = importlib.import_module("diarized_transcriber.transcription.diarization")


def test_assign_speakers_to_segments():
    pipeline = pipeline_mod.DiarizationPipeline(device="cpu")
    speakers = [
        Speaker(id="A", segments=[TimeSegment(start=0.0, end=2.0)]),
        Speaker(id="B", segments=[TimeSegment(start=2.0, end=4.0)])
    ]

    segments = pandas.DataFrame([
        {"start": 0.5, "end": 1.5, "text": "hello"},
        {"start": 2.5, "end": 3.0, "text": "world"}
    ])

    result = pipeline.assign_speakers_to_segments(speakers, segments)
    assert [row.get("speaker") for row in result.data] == ["A", "B"]


def test_assign_speakers_partial_overlap():
    """Segments that partially overlap diarization ranges get the best speaker."""
    pipeline = pipeline_mod.DiarizationPipeline(device="cpu")
    # Speaker A: 0-2s, Speaker B: 2-4s
    speakers = [
        Speaker(id="A", segments=[TimeSegment(start=0.0, end=2.0)]),
        Speaker(id="B", segments=[TimeSegment(start=2.0, end=4.0)])
    ]

    # Segment straddles the A/B boundary but mostly in A (0.8s in A vs 0.2s in B)
    # Old strict-containment would return None; overlap returns "A"
    segments = pandas.DataFrame([
        {"start": 1.2, "end": 2.2, "text": "straddle mostly A"},
    ])

    result = pipeline.assign_speakers_to_segments(speakers, segments)
    assert [row.get("speaker") for row in result.data] == ["A"]


def test_assign_speakers_partial_overlap_favours_larger():
    """When a segment overlaps two speakers, the one with more overlap wins."""
    pipeline = pipeline_mod.DiarizationPipeline(device="cpu")
    speakers = [
        Speaker(id="A", segments=[TimeSegment(start=0.0, end=2.0)]),
        Speaker(id="B", segments=[TimeSegment(start=2.0, end=5.0)])
    ]

    # Segment 1.0-3.0: 1s in A, 1s in B → tie broken by first encountered (A),
    # but let's test a clear winner: 1.5-3.5 → 0.5s in A, 1.5s in B → B wins
    segments = pandas.DataFrame([
        {"start": 1.5, "end": 3.5, "text": "mostly in B"},
    ])

    result = pipeline.assign_speakers_to_segments(speakers, segments)
    assert [row.get("speaker") for row in result.data] == ["B"]


def test_assign_speakers_no_overlap():
    """Segments with no overlapping diarization range get None."""
    pipeline = pipeline_mod.DiarizationPipeline(device="cpu")
    speakers = [
        Speaker(id="A", segments=[TimeSegment(start=0.0, end=1.0)]),
    ]

    # Segment is entirely after speaker A's range
    segments = pandas.DataFrame([
        {"start": 5.0, "end": 6.0, "text": "no overlap"},
    ])

    result = pipeline.assign_speakers_to_segments(speakers, segments)
    assert [row.get("speaker") for row in result.data] == [None]


def test_assign_speakers_multiple_diarization_segments():
    """Speaker with multiple diarization segments; overlap picks the right one."""
    pipeline = pipeline_mod.DiarizationPipeline(device="cpu")
    speakers = [
        Speaker(id="A", segments=[
            TimeSegment(start=0.0, end=1.0),
            TimeSegment(start=3.0, end=5.0),
        ]),
        Speaker(id="B", segments=[TimeSegment(start=1.0, end=3.0)])
    ]

    segments = pandas.DataFrame([
        {"start": 0.2, "end": 0.8, "text": "in A first"},
        {"start": 1.5, "end": 2.5, "text": "in B"},
        {"start": 3.5, "end": 4.5, "text": "in A second"},
    ])

    result = pipeline.assign_speakers_to_segments(speakers, segments)
    assert [row.get("speaker") for row in result.data] == ["A", "B", "A"]


def test_assign_speakers_empty_segments():
    """Empty segment DataFrame returns empty result with speaker column."""
    pipeline = pipeline_mod.DiarizationPipeline(device="cpu")
    speakers = [
        Speaker(id="A", segments=[TimeSegment(start=0.0, end=2.0)]),
    ]

    segments = pandas.DataFrame([])

    result = pipeline.assign_speakers_to_segments(speakers, segments)
    assert result.data == []


def test_default_model_is_updated():
    """Assert default model string is pyannote/speaker-diarization-3.1."""
    pipeline = pipeline_mod.DiarizationPipeline(device="cpu")
    assert pipeline.model == "pyannote/speaker-diarization-3.1"


def test_pipeline_uses_token_kwarg():
    """Verify Pipeline.from_pretrained is called with token= not use_auth_token=."""
    import os
    from unittest.mock import patch, MagicMock

    pipeline = pipeline_mod.DiarizationPipeline(device="cpu")

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.to.return_value = mock_pipeline_instance

    # Patch Pipeline.from_pretrained on the binding inside the diarization module.
    # When other test files (e.g. test_embeddings.py, test_server.py) run first,
    # they may install a different Pipeline class into sys.modules via setdefault
    # or hard assignment. The diarization module captures its Pipeline reference
    # at import time, so we must patch that specific class, not DummyPipeline
    # from this file's module scope.
    actual_pipeline_cls = pipeline_mod.Pipeline
    with patch.dict(os.environ, {"HF_TOKEN": "test-token"}), \
         patch.object(actual_pipeline_cls, "from_pretrained", return_value=mock_pipeline_instance) as mock_from:
        pipeline._initialize_pipeline()
        mock_from.assert_called_once_with(
            "pyannote/speaker-diarization-3.1",
            token="test-token"
        )
