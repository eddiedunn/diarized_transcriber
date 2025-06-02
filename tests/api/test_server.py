import sys
import types
import importlib
from pathlib import Path


def test_transcribe_endpoint():
    # Stub pydantic
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
    pydantic.HttpUrl = str
    sys.modules["pydantic"] = pydantic

    # Stub heavy deps
    torch = types.ModuleType("torch")
    sys.modules["torch"] = torch
    numpy = types.ModuleType("numpy")
    numpy.ndarray = list
    sys.modules["numpy"] = numpy

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
    sys.modules["pandas"] = pandas

    whisperx = types.ModuleType("whisperx")
    sys.modules["whisperx"] = whisperx
    soundfile = types.ModuleType("soundfile")
    soundfile.write = lambda *a, **k: None
    soundfile.read = lambda *a, **k: ([0.0], 16000)
    sys.modules["soundfile"] = soundfile

    pyannote = types.ModuleType("pyannote")
    pyannote_audio = types.ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = type("Pipeline", (), {"from_pretrained": classmethod(lambda cls,*a,**k: cls()), "to": lambda self,d:self})
    pyannote.audio = pyannote_audio
    sys.modules["pyannote"] = pyannote
    sys.modules["pyannote.audio"] = pyannote_audio

    fastapi = types.ModuleType("fastapi")
    class HTTPException(Exception):
        def __init__(self, status_code=500, detail=None):
            self.status_code = status_code
            self.detail = detail
    fastapi.HTTPException = HTTPException
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
    fastapi.FastAPI = FastAPI
    testclient = types.ModuleType("fastapi.testclient")
    class Response:
        def __init__(self, data):
            self.data = data
            self.status_code = 200
        def json(self):
            return self.data
    class TestClient:
        def __init__(self, app):
            self.app = app
        def post(self, path, json=None):
            func = self.app.routes[("POST", path)]
            schemas = importlib.import_module("diarized_transcriber.api.schemas")
            req = schemas.TranscriptionRequest(**(json or {}))
            result = func(req)
            return Response(result.__dict__)
    testclient.TestClient = TestClient
    fastapi.testclient = testclient
    sys.modules["fastapi"] = fastapi
    sys.modules["fastapi.testclient"] = testclient

    pkg = types.ModuleType("diarized_transcriber")
    pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "diarized_transcriber")]
    sys.modules["diarized_transcriber"] = pkg

    engine_mod = importlib.import_module("diarized_transcriber.transcription.engine")
    transcription_mod = importlib.import_module("diarized_transcriber.models.transcription")

    class DummyEngine:
        def __init__(self, *args, model_size="base", **kwargs):
            self.called_with = model_size
        def transcribe(self, content, min_speakers=None, max_speakers=None):
            return transcription_mod.TranscriptionResult(
                content_id=content.id,
                language="en",
                segments=[],
                speakers=None,
                metadata={}
            )

    server = importlib.import_module("diarized_transcriber.api.server")
    schemas = importlib.import_module("diarized_transcriber.api.schemas")
    server.TranscriptionEngine = DummyEngine

    client = testclient.TestClient(server.app)
    resp = client.post(
        "/transcribe",
        json={"id": "1", "title": "Example", "media_url": "http://example.com/a.mp3"},
    )
    assert resp.status_code == 200
    assert resp.json()["result"].content_id == "1"

    # Clean up imported engine module for other tests
    sys.modules.pop("diarized_transcriber.transcription.engine", None)


def test_health_endpoint():
    """Ensure the root endpoint returns a health status."""

    # Reuse the heavy dependency stubs from the previous test
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
    pydantic.HttpUrl = str
    sys.modules["pydantic"] = pydantic

    torch = types.ModuleType("torch")
    sys.modules["torch"] = torch
    numpy = types.ModuleType("numpy")
    numpy.ndarray = list
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
    pyannote_audio.Pipeline = type("Pipeline", (), {"from_pretrained": classmethod(lambda cls, *a, **k: cls()), "to": lambda self, d: self})
    pyannote.audio = pyannote_audio
    sys.modules["pyannote"] = pyannote
    sys.modules["pyannote.audio"] = pyannote_audio

    fastapi = types.ModuleType("fastapi")
    class HTTPException(Exception):
        pass
    fastapi.HTTPException = HTTPException
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
    fastapi.FastAPI = FastAPI
    sys.modules["fastapi"] = fastapi

    pkg = types.ModuleType("diarized_transcriber")
    pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "diarized_transcriber")]
    sys.modules["diarized_transcriber"] = pkg

    server = importlib.import_module("diarized_transcriber.api.server")

    assert server.read_root() == {"status": "ok"}

    sys.modules.pop("diarized_transcriber.api.server", None)
    sys.modules.pop("diarized_transcriber.transcription.engine", None)


def test_server_main_runs_uvicorn():
    """Verify that running the module as __main__ starts uvicorn."""

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = type("BaseModel", (), {})
    pydantic.Field = lambda default=None, **_: default
    pydantic.HttpUrl = str
    sys.modules["pydantic"] = pydantic

    torch = types.ModuleType("torch")
    sys.modules["torch"] = torch
    numpy = types.ModuleType("numpy")
    numpy.ndarray = list
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
    pyannote_audio.Pipeline = type("Pipeline", (), {"from_pretrained": classmethod(lambda cls, *a, **k: cls()), "to": lambda self, d: self})
    pyannote.audio = pyannote_audio
    sys.modules["pyannote"] = pyannote
    sys.modules["pyannote.audio"] = pyannote_audio

    fastapi = types.ModuleType("fastapi")
    class HTTPException(Exception):
        pass
    fastapi.HTTPException = HTTPException
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
    fastapi.FastAPI = FastAPI
    sys.modules["fastapi"] = fastapi

    uvicorn = types.ModuleType("uvicorn")
    def run(app_path, host="0.0.0.0", port=8000, reload=False):
        run.called = {
            "app_path": app_path,
            "host": host,
            "port": port,
            "reload": reload,
        }
    uvicorn.run = run
    sys.modules["uvicorn"] = uvicorn

    pkg = types.ModuleType("diarized_transcriber")
    pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "diarized_transcriber")]
    sys.modules["diarized_transcriber"] = pkg

    import runpy
    sys.modules.pop("diarized_transcriber.api.server", None)
    runpy.run_module("diarized_transcriber.api.server", run_name="__main__")

    assert run.called["app_path"] == "diarized_transcriber.api.server:app"
    assert run.called["port"] == 8000

    sys.modules.pop("diarized_transcriber.api.server", None)
    sys.modules.pop("diarized_transcriber.transcription.engine", None)
    sys.modules.pop("uvicorn", None)
