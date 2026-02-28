import asyncio
import os
import sys
import types
import importlib
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch


def _setup_stubs():
    """Stub heavy dependencies for v1 endpoint tests."""
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
    sys.modules["pydantic"] = pydantic

    pydantic_settings = types.ModuleType("pydantic_settings")

    class BaseSettings:
        model_config = {}

        def __init__(self, **kwargs):
            import typing

            for attr_name in typing.get_type_hints(self.__class__):
                if attr_name == "model_config":
                    continue
                default = getattr(self.__class__, attr_name, None)
                if attr_name in kwargs:
                    default = kwargs[attr_name]
                setattr(self, attr_name, default)

    pydantic_settings.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = pydantic_settings

    torch = types.ModuleType("torch")
    torch.device = lambda x: x
    sys.modules["torch"] = torch

    numpy = types.ModuleType("numpy")
    numpy.ndarray = list
    numpy.float32 = float
    sys.modules["numpy"] = numpy

    pandas = types.ModuleType("pandas")
    pandas.DataFrame = lambda *a, **k: None
    sys.modules["pandas"] = pandas

    whisperx = types.ModuleType("whisperx")
    sys.modules["whisperx"] = whisperx

    soundfile = types.ModuleType("soundfile")
    soundfile.write = lambda *a, **k: None
    soundfile.read = lambda *a, **k: ([0.0], 16000)
    sys.modules["soundfile"] = soundfile

    pyannote = types.ModuleType("pyannote")
    pyannote_audio = types.ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = type(
        "Pipeline",
        (),
        {
            "from_pretrained": classmethod(lambda cls, *a, **k: cls()),
            "to": lambda self, d: self,
        },
    )
    pyannote_audio.Model = type(
        "Model",
        (),
        {"from_pretrained": classmethod(lambda cls, *a, **k: cls())},
    )
    pyannote_audio.Inference = type(
        "Inference",
        (),
        {
            "__init__": lambda self, *a, **k: None,
            "to": lambda self, d: self,
            "crop": lambda self, *a, **k: [0.0] * 256,
        },
    )
    pyannote.audio = pyannote_audio
    pyannote_core = types.ModuleType("pyannote.core")
    pyannote_core.Segment = type(
        "Segment", (), {"__init__": lambda self, start, end: None}
    )
    pyannote.core = pyannote_core
    sys.modules["pyannote"] = pyannote
    sys.modules["pyannote.audio"] = pyannote_audio
    sys.modules["pyannote.core"] = pyannote_core

    pyarrow = types.ModuleType("pyarrow")
    pyarrow.schema = lambda fields: None
    pyarrow.field = lambda name, typ: None
    pyarrow.string = lambda: None
    pyarrow.int64 = lambda: None
    pyarrow.float64 = lambda: None
    pyarrow.float32 = lambda: None
    pyarrow.list_ = lambda typ, size=-1: None
    sys.modules["pyarrow"] = pyarrow

    lancedb_mod = types.ModuleType("lancedb")
    lancedb_mod.connect = lambda path: MagicMock()
    sys.modules["lancedb"] = lancedb_mod

    # Stub fastapi with async support
    fastapi = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code=500, detail=None):
            self.status_code = status_code
            self.detail = detail

    class UploadFile:
        def __init__(self, filename=None, content=b""):
            self.filename = filename
            self._content = content

        async def read(self):
            return self._content

    def Form(default=None):
        return default

    class FastAPI:
        def __init__(self, *a, **k):
            self.routes = {}

        def get(self, path):
            def decorator(fn):
                self.routes[("GET", path)] = fn
                return fn
            return decorator

        def post(self, path, response_model=None):
            def decorator(fn):
                self.routes[("POST", path)] = fn
                return fn
            return decorator

        def include_router(self, router):
            pass

    fastapi.HTTPException = HTTPException
    fastapi.FastAPI = FastAPI
    fastapi.UploadFile = UploadFile
    fastapi.Form = Form
    sys.modules["fastapi"] = fastapi

    return fastapi


def _clear_modules():
    for mod in list(sys.modules):
        if "diarized_transcriber" in mod:
            del sys.modules[mod]


def _setup_package():
    pkg = types.ModuleType("diarized_transcriber")
    pkg.__path__ = [
        str(
            Path(__file__).resolve().parents[2]
            / "src"
            / "diarized_transcriber"
        )
    ]
    sys.modules["diarized_transcriber"] = pkg


def _import_server_with_mock_backend(fastapi_mod):
    """Import server module and inject a mock backend."""
    config_mod = importlib.import_module("diarized_transcriber.config")
    config_mod.get_settings.cache_clear()

    # Import server (backend import will fail gracefully)
    server_mod = importlib.import_module("diarized_transcriber.api.server")

    return server_mod


def _make_dummy_result():
    """Create a dummy TranscriptionResult for testing."""
    transcription_mod = importlib.import_module(
        "diarized_transcriber.models.transcription"
    )
    seg = transcription_mod.TranscriptionSegment(
        start=0.0, end=2.5, text="Hello world", speaker="SPEAKER_00"
    )
    spk = transcription_mod.Speaker(
        id="SPEAKER_00",
        segments=[transcription_mod.TimeSegment(start=0.0, end=2.5)],
        metadata={},
    )
    return transcription_mod.TranscriptionResult(
        content_id="test.wav",
        language="en",
        segments=[seg],
        speakers=[spk],
        metadata={"model_size": "large-v3-turbo", "compute_type": "float16"},
    )


def test_v1_transcribe_response_shape():
    """POST /v1/transcribe returns flat dict with text, segments, language, etc."""
    fastapi_mod = _setup_stubs()
    _clear_modules()
    _setup_package()

    server_mod = _import_server_with_mock_backend(fastapi_mod)
    dummy_result = _make_dummy_result()

    # Inject a mock backend
    mock_backend = MagicMock()
    mock_backend.transcribe = AsyncMock(return_value=dummy_result)
    server_mod._backend = mock_backend

    # Call the endpoint
    upload = fastapi_mod.UploadFile(filename="test.wav", content=b"fake-audio")

    async def run():
        return await server_mod.v1_transcribe(
            audio=upload, language=None, cleanup=False
        )

    result = asyncio.run(run())

    # Verify response shape matches curator's expected fields
    assert "text" in result
    assert "segments" in result
    assert "language" in result
    assert "duration" in result
    assert "model" in result
    assert "speakers" in result

    assert result["text"] == "Hello world"
    assert result["language"] == "en"
    assert result["model"] == "large-v3-turbo"
    assert result["duration"] == 2.5
    assert len(result["segments"]) == 1
    assert result["segments"][0]["speaker"] == "SPEAKER_00"
    assert result["segments"][0]["text"] == "Hello world"
    assert len(result["speakers"]) == 1
    assert result["speakers"][0]["id"] == "SPEAKER_00"
    assert result["speakers"][0]["segment_count"] == 1

    _clear_modules()


def test_v1_transcribe_cleanup_param_accepted():
    """The cleanup param is accepted but has no effect on behavior."""
    fastapi_mod = _setup_stubs()
    _clear_modules()
    _setup_package()

    server_mod = _import_server_with_mock_backend(fastapi_mod)
    dummy_result = _make_dummy_result()

    mock_backend = MagicMock()
    mock_backend.transcribe = AsyncMock(return_value=dummy_result)
    server_mod._backend = mock_backend

    upload = fastapi_mod.UploadFile(filename="test.wav", content=b"fake-audio")

    async def run():
        return await server_mod.v1_transcribe(
            audio=upload, language=None, cleanup=True
        )

    result = asyncio.run(run())
    assert result["text"] == "Hello world"

    _clear_modules()


def test_v1_transcribe_no_backend_returns_503():
    """When backend is None, v1/transcribe returns 503."""
    import pytest

    fastapi_mod = _setup_stubs()
    _clear_modules()
    _setup_package()

    server_mod = _import_server_with_mock_backend(fastapi_mod)
    server_mod._backend = None

    upload = fastapi_mod.UploadFile(filename="test.wav", content=b"fake-audio")

    async def run():
        return await server_mod.v1_transcribe(
            audio=upload, language=None, cleanup=False
        )

    with pytest.raises(fastapi_mod.HTTPException) as exc_info:
        asyncio.run(run())
    assert exc_info.value.status_code == 503

    _clear_modules()


def test_v1_transcribe_temp_file_cleanup_on_error():
    """Temp file is cleaned up even when backend raises."""
    fastapi_mod = _setup_stubs()
    _clear_modules()
    _setup_package()

    server_mod = _import_server_with_mock_backend(fastapi_mod)

    mock_backend = MagicMock()
    mock_backend.transcribe = AsyncMock(
        side_effect=RuntimeError("GPU exploded")
    )
    server_mod._backend = mock_backend

    upload = fastapi_mod.UploadFile(filename="test.wav", content=b"fake-audio")

    created_files = []
    original_named_temp = __import__("tempfile").NamedTemporaryFile

    def tracking_temp(**kwargs):
        f = original_named_temp(**kwargs)
        created_files.append(f.name)
        return f

    import tempfile as tempfile_mod

    original = tempfile_mod.NamedTemporaryFile
    tempfile_mod.NamedTemporaryFile = tracking_temp

    try:

        async def run():
            return await server_mod.v1_transcribe(
                audio=upload, language=None, cleanup=False
            )

        import pytest

        with pytest.raises(fastapi_mod.HTTPException):
            asyncio.run(run())

        # All temp files should have been cleaned up
        for f in created_files:
            assert not os.path.exists(f), f"Temp file {f} was not cleaned up"
    finally:
        tempfile_mod.NamedTemporaryFile = original

    _clear_modules()


def test_health_endpoint():
    """GET /health returns status, model_loaded, gpu_available."""
    fastapi_mod = _setup_stubs()
    _clear_modules()
    _setup_package()

    server_mod = _import_server_with_mock_backend(fastapi_mod)
    # No backend
    server_mod._backend = None

    async def run():
        return await server_mod.health()

    result = asyncio.run(run())
    assert result["status"] == "ok"
    assert result["model_loaded"] is False
    assert result["gpu_available"] is False

    _clear_modules()


def test_models_endpoint():
    """GET /models returns list of available model sizes."""
    fastapi_mod = _setup_stubs()
    _clear_modules()
    _setup_package()

    server_mod = _import_server_with_mock_backend(fastapi_mod)

    async def run():
        return await server_mod.models()

    result = asyncio.run(run())
    assert "models" in result
    assert "large-v3-turbo" in result["models"]
    assert "tiny" in result["models"]
    assert len(result["models"]) == 8

    _clear_modules()
