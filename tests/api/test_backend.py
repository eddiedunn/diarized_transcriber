import asyncio
import sys
import types
import importlib
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock


def _setup_stubs():
    """Stub heavy dependencies for backend tests."""
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

            prefix = ""
            if isinstance(self.__class__.model_config, dict):
                prefix = self.__class__.model_config.get("env_prefix", "")
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

    # Stub gpu_common
    gpu_common = types.ModuleType("gpu_common")

    class GPULock:
        def __init__(self, redis_url):
            self.redis_url = redis_url
            self._acquired = False

        async def connect(self):
            pass

        async def disconnect(self):
            pass

        def acquire(self, timeout=3600, blocking_timeout=300):
            return self

        async def __aenter__(self):
            self._acquired = True
            return self

        async def __aexit__(self, *exc):
            self._acquired = False

    class GPUModelManager:
        def __init__(self, load_fn, idle_timeout=None, cleanup_interval=None):
            self._load_fn = load_fn
            self._model = None

        async def start(self):
            pass

        async def stop(self):
            pass

        async def get(self):
            if self._model is None:
                loop = asyncio.get_running_loop()
                self._model = await loop.run_in_executor(
                    None, self._load_fn
                )
            return self._model

    gpu_common.GPULock = GPULock
    gpu_common.GPUModelManager = GPUModelManager
    sys.modules["gpu_common"] = gpu_common


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


def test_backend_transcribe_acquires_lock_and_runs():
    """Backend.transcribe() acquires the GPU lock and calls engine.transcribe_file()."""
    _setup_stubs()
    _clear_modules()
    _setup_package()

    config_mod = importlib.import_module("diarized_transcriber.config")
    config_mod.get_settings.cache_clear()

    engine_mod = importlib.import_module(
        "diarized_transcriber.transcription.engine"
    )
    engine_mod.verify_gpu_requirements = lambda: "cpu"

    # Mock TranscriptionResult
    transcription_mod = importlib.import_module(
        "diarized_transcriber.models.transcription"
    )
    dummy_result = transcription_mod.TranscriptionResult(
        content_id="test.wav",
        language="en",
        segments=[],
        speakers=None,
        metadata={"model_size": "large-v3-turbo"},
    )

    # Patch engine's transcribe_file to return dummy result
    original_init = engine_mod.TranscriptionEngine.__init__

    def patched_init(self, *args, **kwargs):
        self.model_size = kwargs.get("model_size", "large-v3-turbo")
        self.device = "cpu"

    engine_mod.TranscriptionEngine.__init__ = patched_init
    engine_mod.TranscriptionEngine.transcribe_file = (
        lambda self, path, language=None, min_speakers=None, max_speakers=None: dummy_result
    )

    backend_mod = importlib.import_module("diarized_transcriber.api.backend")

    backend = backend_mod.DiarizedTranscriberBackend()

    async def run():
        result = await backend.transcribe(Path("/tmp/test.wav"))
        return result

    result = asyncio.run(run())
    assert result.content_id == "test.wav"
    assert result.language == "en"

    _clear_modules()


def test_backend_transcribe_error_propagation():
    """Errors from engine.transcribe_file() propagate through the backend."""
    import pytest

    _setup_stubs()
    _clear_modules()
    _setup_package()

    config_mod = importlib.import_module("diarized_transcriber.config")
    config_mod.get_settings.cache_clear()

    engine_mod = importlib.import_module(
        "diarized_transcriber.transcription.engine"
    )
    engine_mod.verify_gpu_requirements = lambda: "cpu"

    def patched_init(self, *args, **kwargs):
        self.model_size = kwargs.get("model_size", "large-v3-turbo")
        self.device = "cpu"

    engine_mod.TranscriptionEngine.__init__ = patched_init

    def exploding_transcribe(self, path, language=None, min_speakers=None, max_speakers=None):
        raise RuntimeError("GPU out of memory")

    engine_mod.TranscriptionEngine.transcribe_file = exploding_transcribe

    backend_mod = importlib.import_module("diarized_transcriber.api.backend")
    backend = backend_mod.DiarizedTranscriberBackend()

    async def run():
        await backend.transcribe(Path("/tmp/test.wav"))

    with pytest.raises(RuntimeError, match="GPU out of memory"):
        asyncio.run(run())

    _clear_modules()
