import sys
import types
import importlib
import math
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Stub minimal pydantic
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

# Stub torch
torch = types.ModuleType("torch")
def device(x):
    return x
torch.device = device
sys.modules.setdefault("torch", torch)

# Functional numpy stub
numpy = types.ModuleType("numpy")

class _ndarray(list):
    """Minimal ndarray stub supporting element-wise ops and tolist."""
    def flatten(self):
        return _ndarray(self)

    def tolist(self):
        return list(self)

    def __truediv__(self, scalar):
        return _ndarray([x / scalar for x in self])

    def __add__(self, other):
        if isinstance(other, (list, _ndarray)):
            return _ndarray([a + b for a, b in zip(self, other)])
        return _ndarray([x + other for x in self])

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

def np_array(data):
    if isinstance(data, _ndarray):
        return data
    return _ndarray(data)

def np_mean(arrays, axis=0):
    if axis == 0 and isinstance(arrays, list) and len(arrays) > 0:
        n = len(arrays)
        dim = len(arrays[0])
        result = []
        for i in range(dim):
            s = sum(arr[i] for arr in arrays)
            result.append(s / n)
        return _ndarray(result)
    return _ndarray(arrays)

class _linalg:
    @staticmethod
    def norm(arr):
        return math.sqrt(sum(x * x for x in arr))

# Ensure numpy module in sys.modules has the attributes we need.
# Another test may have already set a simpler numpy stub via setdefault,
# so we enrich whichever module is present.
if "numpy" in sys.modules:
    numpy = sys.modules["numpy"]
numpy.ndarray = _ndarray
numpy.float32 = float
numpy.array = np_array
numpy.mean = np_mean
numpy.linalg = _linalg
sys.modules["numpy"] = numpy
sys.modules["numpy.linalg"] = _linalg

# Stub pyannote
pyannote = types.ModuleType("pyannote")
pyannote_audio = types.ModuleType("pyannote.audio")
pyannote_core = types.ModuleType("pyannote.core")

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

class DummySegment:
    def __init__(self, start, end):
        self.start = start
        self.end = end

class DummyPipeline:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def to(self, _device):
        return self

pyannote_audio.Pipeline = DummyPipeline
pyannote_audio.Model = DummyModel
pyannote_audio.Inference = DummyInference
pyannote_core.Segment = DummySegment
pyannote.audio = pyannote_audio
pyannote.core = pyannote_core
sys.modules.setdefault("pyannote", pyannote)
sys.modules.setdefault("pyannote.audio", pyannote_audio)
sys.modules.setdefault("pyannote.core", pyannote_core)

# Stub pandas – enrich existing stub if one was already registered
pandas = types.ModuleType("pandas")
if "pandas" in sys.modules:
    pandas = sys.modules["pandas"]
if not hasattr(pandas, "DataFrame"):
    class _Series:
        def __init__(self, data):
            self.data = list(data)
        def __le__(self, other):
            return _Series([x <= other for x in self.data])
        def __ge__(self, other):
            return _Series([x >= other for x in self.data])
        def __and__(self, other):
            return _Series([a and b for a, b in zip(self.data, other.data)])
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
    class _DataFrame:
        def __init__(self, data):
            self.data = [dict(row) for row in data]
        def __getitem__(self, key):
            if isinstance(key, str):
                return _Series([row.get(key) for row in self.data])
            if isinstance(key, _Series):
                filtered = [row for row, flag in zip(self.data, key.data) if flag]
                return _DataFrame(filtered)
            raise TypeError("Unsupported key type")
        def __setitem__(self, key, value):
            values = value.data if isinstance(value, _Series) else value
            for row, val in zip(self.data, values):
                row[key] = val
        def apply(self, func, axis=0):
            assert axis == 1
            return _Series([func(row) for row in self.data])
        @property
        def empty(self):
            return len(self.data) == 0
    pandas.DataFrame = _DataFrame
sys.modules["pandas"] = pandas

# Stub soundfile
soundfile = types.ModuleType("soundfile")
soundfile.write = lambda *a, **k: None
soundfile.read = lambda *a, **k: ([0.0], 16000)
sys.modules.setdefault("soundfile", soundfile)

# Provide lightweight package stub for diarized_transcriber
pkg = types.ModuleType("diarized_transcriber")
pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "diarized_transcriber")]
sys.modules.setdefault("diarized_transcriber", pkg)

# Import modules under test
models = importlib.import_module("diarized_transcriber.models.transcription")
exceptions = importlib.import_module("diarized_transcriber.exceptions")
embeddings_mod = importlib.import_module("diarized_transcriber.transcription.embeddings")

Speaker = models.Speaker
TimeSegment = models.TimeSegment
SpeakerEmbeddingExtractor = embeddings_mod.SpeakerEmbeddingExtractor
DiarizationError = exceptions.DiarizationError
EmbeddingExtractionError = exceptions.EmbeddingExtractionError


def _make_initialized_extractor(mock_inference):
    """Create an extractor with a pre-injected mock inference, bypassing _initialize."""
    extractor = SpeakerEmbeddingExtractor(device="cpu")
    extractor._model = MagicMock()  # non-None so _initialize short-circuits
    extractor._inference = mock_inference
    return extractor


def test_extractor_requires_hf_token():
    """No HF_TOKEN env var raises DiarizationError on _initialize()."""
    import pytest
    extractor = SpeakerEmbeddingExtractor(device="cpu")
    with patch.dict("os.environ", {}, clear=True):
        import os
        env_copy = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with patch.dict("os.environ", env_copy, clear=True):
            with pytest.raises(DiarizationError):
                extractor._initialize()


def test_extractor_initializes_model_once():
    """Call _initialize() twice, Model.from_pretrained called once."""
    extractor = SpeakerEmbeddingExtractor(device="cpu")
    mock_model = MagicMock()
    mock_inference_instance = MagicMock()
    mock_inference_instance.to.return_value = mock_inference_instance

    with patch.dict("os.environ", {"HF_TOKEN": "test-token"}), \
         patch.object(embeddings_mod, "Model", **{"from_pretrained": MagicMock(return_value=mock_model)}) as mock_model_cls, \
         patch.object(embeddings_mod, "Inference", return_value=mock_inference_instance):
        extractor._initialize()
        extractor._initialize()
        mock_model_cls.from_pretrained.assert_called_once()


def test_extract_for_speaker_returns_256_dim():
    """Mock Inference.crop to return a 256-dim array, verify output length."""
    mock_inference = MagicMock()
    mock_inference.crop.return_value = [1.0] * 256
    extractor = _make_initialized_extractor(mock_inference)

    speaker = Speaker(
        id="A",
        segments=[TimeSegment(start=0.0, end=2.0)]
    )
    result = extractor.extract_for_speaker("/tmp/audio.wav", speaker)
    assert len(result) == 256


def test_extract_for_speaker_normalizes_embedding():
    """Verify L2 norm of result is approximately 1.0."""
    mock_inference = MagicMock()
    # Use a non-uniform embedding to verify normalization
    raw_emb = [float(i) for i in range(256)]
    mock_inference.crop.return_value = raw_emb
    extractor = _make_initialized_extractor(mock_inference)

    speaker = Speaker(
        id="A",
        segments=[TimeSegment(start=0.0, end=2.0)]
    )
    result = extractor.extract_for_speaker("/tmp/audio.wav", speaker)
    norm = math.sqrt(sum(x * x for x in result))
    assert abs(norm - 1.0) < 1e-6


def test_extract_for_speaker_averages_segments():
    """3 segments with known embeddings, verify mean is correct."""
    mock_inference = MagicMock()

    # Three different embeddings; each 256-dim
    emb1 = [3.0] * 256
    emb2 = [6.0] * 256
    emb3 = [9.0] * 256
    mock_inference.crop.side_effect = [emb1, emb2, emb3]
    extractor = _make_initialized_extractor(mock_inference)

    speaker = Speaker(
        id="A",
        segments=[
            TimeSegment(start=0.0, end=2.0),
            TimeSegment(start=3.0, end=5.0),
            TimeSegment(start=6.0, end=8.0),
        ]
    )
    result = extractor.extract_for_speaker("/tmp/audio.wav", speaker)
    # Mean of [3, 6, 9] = 6.0 for each dim
    # After L2 normalization, each element = 6.0 / (6.0 * sqrt(256)) = 1/16
    expected_val = 6.0 / (6.0 * math.sqrt(256))
    assert abs(result[0] - expected_val) < 1e-6
    assert abs(result[255] - expected_val) < 1e-6


def test_extract_for_speaker_skips_short_segments():
    """Segment < 0.5s is skipped with logged warning."""
    mock_inference = MagicMock()
    mock_inference.crop.return_value = [1.0] * 256
    extractor = _make_initialized_extractor(mock_inference)

    speaker = Speaker(
        id="A",
        segments=[
            TimeSegment(start=0.0, end=0.3),  # too short (0.3s < 0.5s)
            TimeSegment(start=1.0, end=3.0),   # valid (2.0s)
        ]
    )
    result = extractor.extract_for_speaker("/tmp/audio.wav", speaker)
    # Only the valid segment should be used
    mock_inference.crop.assert_called_once()
    assert len(result) == 256


def test_extract_for_speaker_all_segments_fail_raises():
    """All crops raise -> EmbeddingExtractionError."""
    import pytest
    mock_inference = MagicMock()
    mock_inference.crop.side_effect = RuntimeError("crop failed")
    extractor = _make_initialized_extractor(mock_inference)

    speaker = Speaker(
        id="A",
        segments=[
            TimeSegment(start=0.0, end=2.0),
            TimeSegment(start=3.0, end=5.0),
        ]
    )
    with pytest.raises(EmbeddingExtractionError):
        extractor.extract_for_speaker("/tmp/audio.wav", speaker)


def test_extract_for_speaker_partial_failure_succeeds():
    """1 of 3 fails, still returns embedding from other 2."""
    mock_inference = MagicMock()

    emb1 = [2.0] * 256
    emb3 = [4.0] * 256
    mock_inference.crop.side_effect = [
        emb1,
        RuntimeError("crop failed"),
        emb3,
    ]
    extractor = _make_initialized_extractor(mock_inference)

    speaker = Speaker(
        id="A",
        segments=[
            TimeSegment(start=0.0, end=2.0),
            TimeSegment(start=3.0, end=5.0),
            TimeSegment(start=6.0, end=8.0),
        ]
    )
    result = extractor.extract_for_speaker("/tmp/audio.wav", speaker)
    assert len(result) == 256
    # Mean of [2, 4] = 3.0, then normalized
    norm = math.sqrt(sum(x * x for x in result))
    assert abs(norm - 1.0) < 1e-6


def test_extract_for_speaker_insufficient_speech_raises():
    """Total speech < 1.0s raises EmbeddingExtractionError."""
    import pytest
    mock_inference = MagicMock()
    extractor = _make_initialized_extractor(mock_inference)

    speaker = Speaker(
        id="A",
        segments=[
            TimeSegment(start=0.0, end=0.8),  # 0.8s, passes min segment (0.5s)
        ]
    )
    # Total speech is 0.8s < 1.0s minimum
    with pytest.raises(EmbeddingExtractionError):
        extractor.extract_for_speaker("/tmp/audio.wav", speaker)


def test_extract_for_all_speakers_updates_embeddings():
    """Verify all speakers get embedding field populated."""
    mock_inference = MagicMock()
    mock_inference.crop.return_value = [1.0] * 256
    extractor = _make_initialized_extractor(mock_inference)

    speakers = [
        Speaker(id="A", segments=[TimeSegment(start=0.0, end=2.0)]),
        Speaker(id="B", segments=[TimeSegment(start=3.0, end=5.0)]),
    ]
    result = extractor.extract_for_all_speakers("/tmp/audio.wav", speakers)
    assert len(result) == 2
    for s in result:
        assert s.embedding is not None
        assert len(s.embedding) == 256


def test_extract_for_all_speakers_skips_failed_speakers():
    """One speaker fails, others still get embeddings."""
    mock_inference = MagicMock()
    # Speaker A succeeds, Speaker B fails (all segments fail)
    mock_inference.crop.side_effect = [
        [1.0] * 256,                    # Speaker A succeeds
        RuntimeError("crop failed"),     # Speaker B fails
    ]
    extractor = _make_initialized_extractor(mock_inference)

    speakers = [
        Speaker(id="A", segments=[TimeSegment(start=0.0, end=2.0)]),
        Speaker(id="B", segments=[TimeSegment(start=3.0, end=5.0)]),
    ]
    result = extractor.extract_for_all_speakers("/tmp/audio.wav", speakers)
    assert result[0].embedding is not None
    assert len(result[0].embedding) == 256
    assert result[1].embedding is None


def test_extract_for_speaker_empty_segments_raises():
    """Speaker with no segments raises EmbeddingExtractionError."""
    import pytest
    mock_inference = MagicMock()
    extractor = _make_initialized_extractor(mock_inference)

    speaker = Speaker(id="A", segments=[])
    with pytest.raises(EmbeddingExtractionError):
        extractor.extract_for_speaker("/tmp/audio.wav", speaker)
