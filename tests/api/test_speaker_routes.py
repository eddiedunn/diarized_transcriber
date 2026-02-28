"""Tests for speaker profile management endpoints."""

import sys
import types
import importlib
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# Save a reference to the REAL pydantic and its submodules at import time,
# before other test files (e.g. test_server.py) can replace them with stubs.
# Also save critical attribute references since test_audio.py mutates them
# in-place via sys.modules.setdefault() + attribute overwrites.
import pydantic as _real_pydantic

_real_pydantic_modules = {}
for _key in list(sys.modules.keys()):
    if _key == "pydantic" or _key.startswith("pydantic.") or _key.startswith("pydantic_core"):
        _real_pydantic_modules[_key] = sys.modules[_key]

# Save references to specific pydantic attributes that may be overwritten
_real_pydantic_attrs = {
    "BaseModel": _real_pydantic.BaseModel,
    "Field": _real_pydantic.Field,
    "field_validator": _real_pydantic.field_validator,
    "model_validator": _real_pydantic.model_validator,
}


# ---------------------------------------------------------------------------
# Stub setup helpers
# ---------------------------------------------------------------------------

def _stub_heavy_deps():
    """Stub heavy ML/storage deps so we can import speaker_routes without them."""
    # Stub torch
    torch = types.ModuleType("torch")
    sys.modules["torch"] = torch

    # Stub numpy
    numpy = types.ModuleType("numpy")
    numpy.ndarray = list
    numpy.float32 = float
    sys.modules["numpy"] = numpy

    # Stub pandas
    pandas = types.ModuleType("pandas")
    pandas.DataFrame = lambda *a, **k: None
    sys.modules["pandas"] = pandas

    # Stub whisperx
    whisperx = types.ModuleType("whisperx")
    sys.modules["whisperx"] = whisperx

    # Stub soundfile
    soundfile = types.ModuleType("soundfile")
    soundfile.write = lambda *a, **k: None
    soundfile.read = lambda *a, **k: ([0.0], 16000)
    sys.modules["soundfile"] = soundfile

    # Stub pyannote
    pyannote = types.ModuleType("pyannote")
    pyannote_audio = types.ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = type(
        "Pipeline", (),
        {"from_pretrained": classmethod(lambda cls, *a, **k: cls()),
         "to": lambda self, d: self},
    )
    pyannote_audio.Model = type(
        "Model", (),
        {"from_pretrained": classmethod(lambda cls, *a, **k: cls())},
    )
    pyannote_audio.Inference = type(
        "Inference", (),
        {"__init__": lambda self, model, window=None: None,
         "crop": lambda self, file, segment: [0.0] * 256,
         "to": lambda self, device: self},
    )
    pyannote.audio = pyannote_audio
    pyannote_core = types.ModuleType("pyannote.core")
    pyannote_core.Segment = type(
        "Segment", (),
        {"__init__": lambda self, start, end:
         setattr(self, "start", start) or setattr(self, "end", end)},
    )
    pyannote.core = pyannote_core
    sys.modules["pyannote"] = pyannote
    sys.modules["pyannote.audio"] = pyannote_audio
    sys.modules["pyannote.core"] = pyannote_core

    # Stub pyarrow
    pyarrow = types.ModuleType("pyarrow")
    pyarrow.schema = lambda fields: None
    pyarrow.field = lambda name, typ: None
    pyarrow.string = lambda: None
    pyarrow.int64 = lambda: None
    pyarrow.float64 = lambda: None
    pyarrow.float32 = lambda: None
    pyarrow.list_ = lambda typ, size=-1: None
    sys.modules["pyarrow"] = pyarrow

    # Stub lancedb
    lancedb = types.ModuleType("lancedb")
    lancedb.connect = lambda path: MagicMock()
    sys.modules["lancedb"] = lancedb

    # Stub fastapi with real-enough HTTPException and APIRouter
    fastapi = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code=500, detail=None):
            self.status_code = status_code
            self.detail = detail

    class APIRouter:
        def __init__(self, prefix="", tags=None):
            self.prefix = prefix
            self.tags = tags or []
            self.routes = {}

        def get(self, path, response_model=None, status_code=200):
            def decorator(fn):
                self.routes[("GET", path)] = fn
                return fn
            return decorator

        def post(self, path, response_model=None, status_code=200):
            def decorator(fn):
                self.routes[("POST", path)] = fn
                return fn
            return decorator

        def put(self, path, response_model=None, status_code=200):
            def decorator(fn):
                self.routes[("PUT", path)] = fn
                return fn
            return decorator

        def delete(self, path, status_code=200):
            def decorator(fn):
                self.routes[("DELETE", path)] = fn
                return fn
            return decorator

    class FastAPI:
        def __init__(self, *a, **k):
            self.routes = {}
            self._routers = []

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
            self._routers.append(router)

    fastapi.HTTPException = HTTPException
    fastapi.FastAPI = FastAPI
    fastapi.APIRouter = APIRouter
    sys.modules["fastapi"] = fastapi

    return HTTPException


def _setup_package():
    """Set up the diarized_transcriber package stub."""
    pkg = types.ModuleType("diarized_transcriber")
    pkg.__path__ = [
        str(Path(__file__).resolve().parents[2] / "src" / "diarized_transcriber")
    ]
    sys.modules["diarized_transcriber"] = pkg


def _clear_modules():
    """Clear cached modules so reimport picks up stubs."""
    mods = [
        "diarized_transcriber.api",
        "diarized_transcriber.api.schemas",
        "diarized_transcriber.api.speaker_schemas",
        "diarized_transcriber.api.server",
        "diarized_transcriber.api.speaker_routes",
        "diarized_transcriber.exceptions",
        "diarized_transcriber.models",
        "diarized_transcriber.models.content",
        "diarized_transcriber.models.transcription",
        "diarized_transcriber.models.speaker_profile",
        "diarized_transcriber.storage",
        "diarized_transcriber.storage.speaker_store",
        "diarized_transcriber.identification",
        "diarized_transcriber.identification.matcher",
        "diarized_transcriber.transcription",
        "diarized_transcriber.transcription.engine",
        "diarized_transcriber.transcription.audio",
        "diarized_transcriber.transcription.diarization",
        "diarized_transcriber.transcription.embeddings",
        "diarized_transcriber.transcription.gpu",
        "diarized_transcriber.types",
        "diarized_transcriber.utils",
    ]
    for mod in mods:
        sys.modules.pop(mod, None)


def _restore_real_pydantic():
    """Ensure the real pydantic module is available (undo stubs from other tests).

    Some test files (e.g. test_audio.py) use sys.modules.setdefault() which
    returns the existing real pydantic, then overwrite attributes (BaseModel,
    Field) on the real module object.  A simple sys.modules restore cannot undo
    those in-place mutations, so we also restore the saved attribute references.
    """
    # Remove any stub or corrupted pydantic modules
    keys_to_remove = [
        k for k in sys.modules
        if k == "pydantic" or k.startswith("pydantic.") or k.startswith("pydantic_core")
    ]
    for key in keys_to_remove:
        sys.modules.pop(key, None)
    # Restore saved module references
    sys.modules.update(_real_pydantic_modules)
    # Restore in-place attribute mutations on the real module
    for attr_name, attr_val in _real_pydantic_attrs.items():
        setattr(_real_pydantic, attr_name, attr_val)


def _get_routes_and_deps():
    """Import speaker_routes and related modules after stubbing."""
    _restore_real_pydantic()
    HTTPException = _stub_heavy_deps()
    _setup_package()
    _clear_modules()

    # Import the modules we need
    exceptions_mod = importlib.import_module("diarized_transcriber.exceptions")
    profile_mod = importlib.import_module("diarized_transcriber.models.speaker_profile")
    schemas_mod = importlib.import_module("diarized_transcriber.api.speaker_schemas")
    routes_mod = importlib.import_module("diarized_transcriber.api.speaker_routes")

    return routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException


def _make_embedding(value=0.0625):
    """Create a 256-dim embedding with uniform value."""
    return [value] * 256


def _make_mock_profile(profile_mod, profile_id="prof-1", name=None,
                       embedding=None, enrollment_count=1,
                       total_duration=5.0):
    """Create a SpeakerProfile instance."""
    if embedding is None:
        embedding = _make_embedding()
    return profile_mod.SpeakerProfile(
        id=profile_id,
        name=name,
        embedding=embedding,
        enrollment_count=enrollment_count,
        total_duration=total_duration,
    )


# ---------------------------------------------------------------------------
# 1. test_list_profiles_empty
# ---------------------------------------------------------------------------

def test_list_profiles_empty():
    """GET /speakers/ with no profiles returns empty list."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    mock_store = MagicMock()
    mock_store.list_profiles.return_value = []

    with patch.object(routes_mod, "get_store", return_value=mock_store):
        result = routes_mod.list_profiles()

    assert result.count == 0
    assert result.profiles == []


# ---------------------------------------------------------------------------
# 2. test_list_profiles_with_data
# ---------------------------------------------------------------------------

def test_list_profiles_with_data():
    """GET /speakers/ returns all stored profiles."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    p1 = _make_mock_profile(profile_mod, profile_id="p1", name="Alice")
    p2 = _make_mock_profile(profile_mod, profile_id="p2", name="Bob")

    mock_store = MagicMock()
    mock_store.list_profiles.return_value = [p1, p2]

    with patch.object(routes_mod, "get_store", return_value=mock_store):
        result = routes_mod.list_profiles()

    assert result.count == 2
    assert len(result.profiles) == 2
    assert result.profiles[0].id == "p1"
    assert result.profiles[1].id == "p2"


# ---------------------------------------------------------------------------
# 3. test_get_profile_found
# ---------------------------------------------------------------------------

def test_get_profile_found():
    """GET /speakers/{id} returns the profile when it exists."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    profile = _make_mock_profile(profile_mod, profile_id="p1", name="Alice")
    mock_store = MagicMock()
    mock_store.get_profile.return_value = profile

    with patch.object(routes_mod, "get_store", return_value=mock_store):
        result = routes_mod.get_profile("p1")

    assert result.id == "p1"
    assert result.name == "Alice"


# ---------------------------------------------------------------------------
# 4. test_get_profile_not_found
# ---------------------------------------------------------------------------

def test_get_profile_not_found():
    """GET /speakers/{id} returns 404 when profile does not exist."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    mock_store = MagicMock()
    mock_store.get_profile.return_value = None

    with patch.object(routes_mod, "get_store", return_value=mock_store):
        with pytest.raises(HTTPException) as exc_info:
            routes_mod.get_profile("nonexistent")

    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# 5. test_enroll_speaker_success
# ---------------------------------------------------------------------------

def test_enroll_speaker_success():
    """POST /speakers/enroll with valid embedding creates a profile."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    embedding = _make_embedding()
    req = schemas_mod.EnrollSpeakerRequest(embedding=embedding)

    mock_store = MagicMock()
    mock_store.add_profile.side_effect = lambda p: p

    with patch.object(routes_mod, "get_store", return_value=mock_store):
        result = routes_mod.enroll_speaker(req)

    assert result.embedding == embedding
    assert result.name is None
    mock_store.add_profile.assert_called_once()


# ---------------------------------------------------------------------------
# 6. test_enroll_speaker_with_name
# ---------------------------------------------------------------------------

def test_enroll_speaker_with_name():
    """POST /speakers/enroll with name sets the profile name."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    embedding = _make_embedding()
    req = schemas_mod.EnrollSpeakerRequest(embedding=embedding, name="Alice")

    mock_store = MagicMock()
    mock_store.add_profile.side_effect = lambda p: p

    with patch.object(routes_mod, "get_store", return_value=mock_store):
        result = routes_mod.enroll_speaker(req)

    assert result.name == "Alice"


# ---------------------------------------------------------------------------
# 7. test_enroll_speaker_wrong_embedding_dim
# ---------------------------------------------------------------------------

def test_enroll_speaker_wrong_embedding_dim():
    """POST /speakers/enroll with wrong embedding dimension raises ValidationError."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        schemas_mod.EnrollSpeakerRequest(embedding=[0.1] * 128)


# ---------------------------------------------------------------------------
# 8. test_enroll_speaker_empty_embedding
# ---------------------------------------------------------------------------

def test_enroll_speaker_empty_embedding():
    """POST /speakers/enroll with empty embedding raises ValidationError."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        schemas_mod.EnrollSpeakerRequest(embedding=[])


# ---------------------------------------------------------------------------
# 9. test_enroll_speaker_non_numeric_embedding
# ---------------------------------------------------------------------------

def test_enroll_speaker_non_numeric_embedding():
    """POST /speakers/enroll with non-numeric embedding raises ValidationError."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        schemas_mod.EnrollSpeakerRequest(embedding=["a"] * 256)


# ---------------------------------------------------------------------------
# 10. test_name_speaker_success
# ---------------------------------------------------------------------------

def test_name_speaker_success():
    """PUT /speakers/{id}/name renames the profile."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    profile = _make_mock_profile(profile_mod, profile_id="p1", name="OldName")
    req = schemas_mod.NameSpeakerRequest(name="NewName")

    mock_store = MagicMock()
    mock_store.get_profile.return_value = profile
    mock_store.update_profile.side_effect = lambda p: p

    with patch.object(routes_mod, "get_store", return_value=mock_store):
        result = routes_mod.name_speaker("p1", req)

    assert result.name == "NewName"
    mock_store.update_profile.assert_called_once()


# ---------------------------------------------------------------------------
# 11. test_name_speaker_not_found
# ---------------------------------------------------------------------------

def test_name_speaker_not_found():
    """PUT /speakers/{id}/name returns 404 when profile does not exist."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    req = schemas_mod.NameSpeakerRequest(name="NewName")

    mock_store = MagicMock()
    mock_store.get_profile.return_value = None

    with patch.object(routes_mod, "get_store", return_value=mock_store):
        with pytest.raises(HTTPException) as exc_info:
            routes_mod.name_speaker("nonexistent", req)

    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# 12. test_name_speaker_empty_name
# ---------------------------------------------------------------------------

def test_name_speaker_empty_name():
    """PUT /speakers/{id}/name with empty name raises ValidationError."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        schemas_mod.NameSpeakerRequest(name="")


# ---------------------------------------------------------------------------
# 13. test_name_speaker_too_long
# ---------------------------------------------------------------------------

def test_name_speaker_too_long():
    """PUT /speakers/{id}/name with name > 200 chars raises ValidationError."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        schemas_mod.NameSpeakerRequest(name="x" * 201)


# ---------------------------------------------------------------------------
# 14. test_name_speaker_strips_whitespace
# ---------------------------------------------------------------------------

def test_name_speaker_strips_whitespace():
    """PUT /speakers/{id}/name strips leading/trailing whitespace."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    req = schemas_mod.NameSpeakerRequest(name="  Alice  ")
    assert req.name == "Alice"


# ---------------------------------------------------------------------------
# 15. test_delete_profile_success
# ---------------------------------------------------------------------------

def test_delete_profile_success():
    """DELETE /speakers/{id} deletes the profile."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    profile = _make_mock_profile(profile_mod, profile_id="p1")
    mock_store = MagicMock()
    mock_store.get_profile.return_value = profile

    with patch.object(routes_mod, "get_store", return_value=mock_store):
        result = routes_mod.delete_profile("p1")

    assert result == {"detail": "Profile deleted"}
    mock_store.delete_profile.assert_called_once_with("p1")


# ---------------------------------------------------------------------------
# 16. test_delete_profile_not_found
# ---------------------------------------------------------------------------

def test_delete_profile_not_found():
    """DELETE /speakers/{id} returns 404 when profile does not exist."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    mock_store = MagicMock()
    mock_store.get_profile.return_value = None

    with patch.object(routes_mod, "get_store", return_value=mock_store):
        with pytest.raises(HTTPException) as exc_info:
            routes_mod.delete_profile("nonexistent")

    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# 17. test_search_speakers_returns_matches
# ---------------------------------------------------------------------------

def test_search_speakers_returns_matches():
    """POST /speakers/search returns matching profiles."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    embedding = _make_embedding()
    req = schemas_mod.SearchSpeakersRequest(embedding=embedding)

    match = profile_mod.SpeakerMatch(
        profile_id="p1", distance=0.1, confidence=0.9
    )
    profile = _make_mock_profile(profile_mod, profile_id="p1", name="Alice")

    mock_store = MagicMock()
    mock_store.search.return_value = [match]
    mock_store.get_profile.return_value = profile

    with patch.object(routes_mod, "get_store", return_value=mock_store):
        result = routes_mod.search_speakers(req)

    assert result.count == 1
    assert result.matches[0]["profile_id"] == "p1"
    assert result.matches[0]["distance"] == 0.1


# ---------------------------------------------------------------------------
# 18. test_search_speakers_no_matches
# ---------------------------------------------------------------------------

def test_search_speakers_no_matches():
    """POST /speakers/search returns empty when no matches found."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    embedding = _make_embedding()
    req = schemas_mod.SearchSpeakersRequest(embedding=embedding)

    mock_store = MagicMock()
    mock_store.search.return_value = []

    with patch.object(routes_mod, "get_store", return_value=mock_store):
        result = routes_mod.search_speakers(req)

    assert result.count == 0
    assert result.matches == []


# ---------------------------------------------------------------------------
# 19. test_search_speakers_wrong_embedding_dim
# ---------------------------------------------------------------------------

def test_search_speakers_wrong_embedding_dim():
    """POST /speakers/search with wrong embedding dim raises ValidationError."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        schemas_mod.SearchSpeakersRequest(embedding=[0.1] * 128)


# ---------------------------------------------------------------------------
# 20. test_search_speakers_limit_validation
# ---------------------------------------------------------------------------

def test_search_speakers_limit_validation():
    """POST /speakers/search with limit out of range raises ValidationError."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    from pydantic import ValidationError

    # limit = 0 (below minimum of 1)
    with pytest.raises(ValidationError):
        schemas_mod.SearchSpeakersRequest(
            embedding=_make_embedding(), limit=0
        )

    # limit = 51 (above maximum of 50)
    with pytest.raises(ValidationError):
        schemas_mod.SearchSpeakersRequest(
            embedding=_make_embedding(), limit=51
        )


# ---------------------------------------------------------------------------
# 21. test_search_speakers_threshold_validation
# ---------------------------------------------------------------------------

def test_search_speakers_threshold_validation():
    """POST /speakers/search with threshold out of range raises ValidationError."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    from pydantic import ValidationError

    # threshold = -0.1 (below minimum)
    with pytest.raises(ValidationError):
        schemas_mod.SearchSpeakersRequest(
            embedding=_make_embedding(), threshold=-0.1
        )

    # threshold = 2.1 (above maximum)
    with pytest.raises(ValidationError):
        schemas_mod.SearchSpeakersRequest(
            embedding=_make_embedding(), threshold=2.1
        )


# ---------------------------------------------------------------------------
# 22. test_merge_profiles_success
# ---------------------------------------------------------------------------

def test_merge_profiles_success():
    """POST /speakers/merge merges source into target."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    req = schemas_mod.MergeProfilesRequest(
        source_profile_id="src-1", target_profile_id="tgt-1"
    )

    merged_profile = _make_mock_profile(
        profile_mod, profile_id="tgt-1", name="Merged",
        enrollment_count=3
    )

    mock_store = MagicMock()
    mock_store.merge_profiles.return_value = merged_profile

    with patch.object(routes_mod, "get_store", return_value=mock_store):
        result = routes_mod.merge_profiles(req)

    assert result.id == "tgt-1"
    assert result.enrollment_count == 3
    mock_store.merge_profiles.assert_called_once_with(
        source_id="src-1", target_id="tgt-1"
    )


# ---------------------------------------------------------------------------
# 23. test_merge_profiles_same_id
# ---------------------------------------------------------------------------

def test_merge_profiles_same_id():
    """POST /speakers/merge with same source and target raises ValidationError."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        schemas_mod.MergeProfilesRequest(
            source_profile_id="same-id", target_profile_id="same-id"
        )


# ---------------------------------------------------------------------------
# 24. test_merge_profiles_source_not_found
# ---------------------------------------------------------------------------

def test_merge_profiles_source_not_found():
    """POST /speakers/merge returns 400 when source profile not found."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    StorageError = exceptions_mod.StorageError

    req = schemas_mod.MergeProfilesRequest(
        source_profile_id="missing", target_profile_id="tgt-1"
    )

    mock_store = MagicMock()
    mock_store.merge_profiles.side_effect = StorageError(
        "Source profile missing not found"
    )

    with patch.object(routes_mod, "get_store", return_value=mock_store):
        with pytest.raises(HTTPException) as exc_info:
            routes_mod.merge_profiles(req)

    assert exc_info.value.status_code == 400
    assert "not found" in exc_info.value.detail


# ---------------------------------------------------------------------------
# 25. test_merge_profiles_target_not_found
# ---------------------------------------------------------------------------

def test_merge_profiles_target_not_found():
    """POST /speakers/merge returns 400 when target profile not found."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    StorageError = exceptions_mod.StorageError

    req = schemas_mod.MergeProfilesRequest(
        source_profile_id="src-1", target_profile_id="missing"
    )

    mock_store = MagicMock()
    mock_store.merge_profiles.side_effect = StorageError(
        "Target profile missing not found"
    )

    with patch.object(routes_mod, "get_store", return_value=mock_store):
        with pytest.raises(HTTPException) as exc_info:
            routes_mod.merge_profiles(req)

    assert exc_info.value.status_code == 400
    assert "not found" in exc_info.value.detail


# ---------------------------------------------------------------------------
# 26. test_transcribe_response_includes_embeddings
# ---------------------------------------------------------------------------

def test_transcribe_response_includes_embeddings():
    """TranscriptionResponse can include speaker embeddings."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    transcription_mod = importlib.import_module(
        "diarized_transcriber.models.transcription"
    )
    original_schemas = importlib.import_module(
        "diarized_transcriber.api.schemas"
    )

    embedding = _make_embedding()
    speaker = transcription_mod.Speaker(
        id="SPEAKER_00",
        segments=[transcription_mod.TimeSegment(start=0.0, end=5.0)],
        embedding=embedding,
    )
    result = transcription_mod.TranscriptionResult(
        content_id="test-1",
        language="en",
        segments=[],
        speakers=[speaker],
        metadata={},
    )

    response = original_schemas.TranscriptionResponse(result=result)
    assert response.result.speakers[0].embedding == embedding


# ---------------------------------------------------------------------------
# 27. test_transcribe_response_includes_identified_names
# ---------------------------------------------------------------------------

def test_transcribe_response_includes_identified_names():
    """TranscriptionResponse can include speaker metadata with identified names."""
    routes_mod, schemas_mod, profile_mod, exceptions_mod, HTTPException = (
        _get_routes_and_deps()
    )

    transcription_mod = importlib.import_module(
        "diarized_transcriber.models.transcription"
    )
    original_schemas = importlib.import_module(
        "diarized_transcriber.api.schemas"
    )

    embedding = _make_embedding()
    speaker = transcription_mod.Speaker(
        id="SPEAKER_00",
        segments=[transcription_mod.TimeSegment(start=0.0, end=5.0)],
        embedding=embedding,
        metadata={"identified_name": "Alice", "profile_id": "prof-1"},
    )
    result = transcription_mod.TranscriptionResult(
        content_id="test-1",
        language="en",
        segments=[],
        speakers=[speaker],
        metadata={},
    )

    response = original_schemas.TranscriptionResponse(result=result)
    assert response.result.speakers[0].metadata["identified_name"] == "Alice"
    assert response.result.speakers[0].metadata["profile_id"] == "prof-1"
