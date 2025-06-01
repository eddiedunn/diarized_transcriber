import sys
import types
import importlib
from pathlib import Path
import os

# Minimal numpy stub for type imports
numpy = sys.modules.setdefault("numpy", types.ModuleType("numpy"))
numpy.ndarray = list

# Stub minimal pydantic so modules can import without dependency
pydantic = sys.modules.setdefault("pydantic", types.ModuleType("pydantic"))
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

# Stub soundfile with minimal read/write
soundfile = sys.modules.setdefault("soundfile", types.ModuleType("soundfile"))

def dummy_write(path, data, samplerate):
    with open(path, "wb") as _f:
        pass

def dummy_read(path):
    return [0.0], 16000

soundfile.write = dummy_write
soundfile.read = dummy_read

# Stub whisperx so audio module imports cleanly
whisperx = sys.modules.setdefault("whisperx", types.ModuleType("whisperx"))
whisperx.load_audio = lambda x: [0.0]

# Provide lightweight package stub for diarized_transcriber
pkg = types.ModuleType("diarized_transcriber")
pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "diarized_transcriber")]
sys.modules.setdefault("diarized_transcriber", pkg)

# Import modules under test
content_mod = importlib.import_module("diarized_transcriber.models.content")
audio_mod = importlib.import_module("diarized_transcriber.transcription.audio")
MediaContent = content_mod.MediaContent
MediaSource = content_mod.MediaSource


def test_create_temp_audio_file():
    calls = {}

    def mock_write(path, data, sr):
        calls["args"] = (path, data, sr)
        with open(path, "wb"):
            pass

    audio_mod.sf.write = mock_write

    data = [0.0, 1.0]
    path, tmp = audio_mod.create_temp_audio_file(data, 16000)

    assert os.path.exists(path)
    assert calls["args"][0] == path
    assert calls["args"][1] == data
    assert calls["args"][2] == 16000

    tmp.close()
    os.unlink(path)
    assert not os.path.exists(path)


def test_process_media_content_invokes_load_audio():
    called = {}

    def mock_load_audio(src):
        called["src"] = src
        return [0.0], 16000

    audio_mod.load_audio = mock_load_audio

    content = MediaContent(
        id="1",
        title="t",
        media_url="http://example.com/audio.wav",
        source=MediaSource(type="podcast")
    )

    audio_mod.process_media_content(content)
    assert called["src"] == content.media_url
