import sys
import types
import importlib
from pathlib import Path
from unittest.mock import patch
import pytest

# Stub minimal pydantic so module imports succeed
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

# Stub torch with minimal cuda features
torch = sys.modules.setdefault("torch", types.ModuleType("torch"))

if not hasattr(torch, "device"):
    def device(x):
        return x
    torch.device = device

torch.__version__ = getattr(torch, "__version__", "1.0")
torch.version = getattr(torch, "version", types.SimpleNamespace())
torch.version.cuda = "11.0"

class DummyProps:
    total_memory = 16 * 1024 * 1024 * 1024

torch.cuda = getattr(torch, "cuda", types.SimpleNamespace())
torch.cuda.is_available = lambda: True
torch.cuda.empty_cache = lambda: None
torch.cuda.get_device_name = lambda idx: "Fake GPU"
torch.cuda.get_device_properties = lambda idx: DummyProps()


# Provide package stub for diarized_transcriber
pkg = types.ModuleType("diarized_transcriber")
pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "diarized_transcriber")]
sys.modules.setdefault("diarized_transcriber", pkg)

# Import modules under test
gpu = importlib.import_module("diarized_transcriber.transcription.gpu")
GPUConfigError = importlib.import_module("diarized_transcriber.exceptions").GPUConfigError


def test_verify_gpu_requirements_unavailable():
    with patch.object(torch.cuda, "is_available", return_value=False):
        with pytest.raises(GPUConfigError):
            gpu.verify_gpu_requirements()


def test_verify_gpu_requirements_available():
    with patch.object(torch.cuda, "is_available", return_value=True):
        assert gpu.verify_gpu_requirements() == "cuda"


def test_cleanup_gpu_memory_calls_torch_and_gc():
    with patch.object(torch.cuda, "empty_cache") as empty_cache, patch("gc.collect") as gc_collect:
        gpu.cleanup_gpu_memory()
        empty_cache.assert_called_once_with()
        gc_collect.assert_called_once_with()
