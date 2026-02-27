"""Tests for SpeakerIdentifier (cross-recording speaker identification)."""

import math
import sys
import types
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch


def _setup_stubs():
    """Set up module stubs for identification tests."""
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

    numpy = types.ModuleType("numpy")
    numpy.ndarray = list
    numpy.float32 = float
    sys.modules["numpy"] = numpy

    pyarrow = types.ModuleType("pyarrow")
    pyarrow.schema = lambda fields: None
    pyarrow.field = lambda name, typ: None
    pyarrow.string = lambda: None
    pyarrow.int64 = lambda: None
    pyarrow.float64 = lambda: None
    pyarrow.float32 = lambda: None
    pyarrow.list_ = lambda typ, size=-1: None
    sys.modules["pyarrow"] = pyarrow

    lancedb = types.ModuleType("lancedb")
    lancedb.connect = lambda path: MagicMock()
    sys.modules["lancedb"] = lancedb

    pyannote = types.ModuleType("pyannote")
    pyannote_audio = types.ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = type("Pipeline", (), {
        "from_pretrained": classmethod(lambda cls, *a, **k: cls()),
    })
    pyannote_audio.Model = type("Model", (), {
        "from_pretrained": classmethod(lambda cls, *a, **k: cls()),
    })
    pyannote_audio.Inference = type("Inference", (), {
        "__init__": lambda self, *a, **k: None,
    })
    pyannote.audio = pyannote_audio
    pyannote_core = types.ModuleType("pyannote.core")
    pyannote_core.Segment = type("Segment", (), {
        "__init__": lambda self, start, end: None,
    })
    pyannote.core = pyannote_core
    sys.modules["pyannote"] = pyannote
    sys.modules["pyannote.audio"] = pyannote_audio
    sys.modules["pyannote.core"] = pyannote_core

    pkg = types.ModuleType("diarized_transcriber")
    pkg.__path__ = [
        str(Path(__file__).resolve().parents[2] / "src" / "diarized_transcriber")
    ]
    sys.modules["diarized_transcriber"] = pkg


def _force_reimport():
    """Clear cached module imports."""
    mods_to_clear = [
        "diarized_transcriber.identification",
        "diarized_transcriber.identification.matcher",
        "diarized_transcriber.storage",
        "diarized_transcriber.storage.speaker_store",
        "diarized_transcriber.models.transcription",
        "diarized_transcriber.models.speaker_profile",
        "diarized_transcriber.exceptions",
        "diarized_transcriber.transcription.embeddings",
        "diarized_transcriber.transcription.diarization",
        "diarized_transcriber.transcription.gpu",
        "diarized_transcriber.transcription.audio",
        "diarized_transcriber.types",
    ]
    for mod in mods_to_clear:
        sys.modules.pop(mod, None)


def _get_modules():
    """Import and return the required modules after stubs are set up."""
    _setup_stubs()
    _force_reimport()
    matcher_mod = importlib.import_module(
        "diarized_transcriber.identification.matcher"
    )
    transcription_mod = importlib.import_module(
        "diarized_transcriber.models.transcription"
    )
    profile_mod = importlib.import_module(
        "diarized_transcriber.models.speaker_profile"
    )
    return matcher_mod, transcription_mod, profile_mod


def _make_speaker(transcription_mod, speaker_id, embedding=None, segments=None):
    """Create a Speaker object with optional embedding and segments."""
    TimeSegment = transcription_mod.TimeSegment
    if segments is None:
        segments = [TimeSegment(start=0.0, end=5.0)]
    return transcription_mod.Speaker(
        id=speaker_id, segments=segments, embedding=embedding
    )


def _make_profile(profile_mod, profile_id="prof-1", name=None, embedding=None,
                  enrollment_count=1, total_duration=5.0):
    """Create a SpeakerProfile."""
    if embedding is None:
        embedding = [0.1] * 256
    return profile_mod.SpeakerProfile(
        id=profile_id,
        name=name,
        embedding=embedding,
        enrollment_count=enrollment_count,
        total_duration=total_duration,
    )


def _make_match(profile_mod, profile_id="prof-1", distance=0.1):
    """Create a SpeakerMatch."""
    return profile_mod.SpeakerMatch(
        profile_id=profile_id,
        distance=distance,
        confidence=1.0 - distance,
    )


# --- Tests ---


def test_identify_known_speaker():
    """Store returns match above threshold -> speaker matched."""
    matcher_mod, transcription_mod, profile_mod = _get_modules()

    embedding = [0.1] * 256
    speaker = _make_speaker(transcription_mod, "SPEAKER_00", embedding=embedding)
    profile = _make_profile(profile_mod, profile_id="known-1", name="Alice")
    match = _make_match(profile_mod, profile_id="known-1", distance=0.15)

    mock_store = MagicMock()
    mock_store.search.return_value = [match]
    mock_store.get_profile.return_value = profile

    identifier = matcher_mod.SpeakerIdentifier(store=mock_store, match_threshold=0.4)
    results = identifier.identify_speakers([speaker])

    assert len(results) == 1
    assert results[0][0] is speaker
    assert results[0][1] is profile
    mock_store.search.assert_called_once()


def test_identify_unknown_speaker_no_auto_enroll():
    """No match, auto_enroll=False -> (speaker, None)."""
    matcher_mod, transcription_mod, profile_mod = _get_modules()

    embedding = [0.2] * 256
    speaker = _make_speaker(transcription_mod, "SPEAKER_01", embedding=embedding)

    mock_store = MagicMock()
    mock_store.search.return_value = []

    identifier = matcher_mod.SpeakerIdentifier(
        store=mock_store, match_threshold=0.4, auto_enroll=False
    )
    results = identifier.identify_speakers([speaker])

    assert len(results) == 1
    assert results[0][0] is speaker
    assert results[0][1] is None


def test_identify_unknown_speaker_auto_enroll():
    """No match, auto_enroll=True -> new profile created and returned."""
    matcher_mod, transcription_mod, profile_mod = _get_modules()

    embedding = [0.3] * 256
    speaker = _make_speaker(transcription_mod, "SPEAKER_02", embedding=embedding)

    mock_store = MagicMock()
    mock_store.search.return_value = []
    mock_store.add_profile.side_effect = lambda p: p

    identifier = matcher_mod.SpeakerIdentifier(
        store=mock_store, match_threshold=0.4, auto_enroll=True
    )
    results = identifier.identify_speakers([speaker])

    assert len(results) == 1
    assert results[0][0] is speaker
    assert results[0][1] is not None  # profile was returned
    # Profile was created (add_profile called)
    mock_store.add_profile.assert_called_once()


def test_identify_speaker_without_embedding_skips():
    """Embedding=None -> (speaker, None), warning logged."""
    import logging

    matcher_mod, transcription_mod, profile_mod = _get_modules()

    speaker = _make_speaker(transcription_mod, "SPEAKER_03", embedding=None)

    mock_store = MagicMock()

    identifier = matcher_mod.SpeakerIdentifier(store=mock_store, match_threshold=0.4)

    with patch.object(matcher_mod.logger, "warning") as mock_warn:
        results = identifier.identify_speakers([speaker])

    assert len(results) == 1
    assert results[0][0] is speaker
    assert results[0][1] is None
    # Store.search should NOT have been called
    mock_store.search.assert_not_called()
    # Warning should have been logged with speaker ID
    mock_warn.assert_called_once()
    assert "SPEAKER_03" in str(mock_warn.call_args)


def test_identify_multiple_speakers_mixed():
    """3 speakers: 1 matched, 1 unknown, 1 no embedding."""
    matcher_mod, transcription_mod, profile_mod = _get_modules()

    emb_known = [0.1] * 256
    emb_unknown = [0.9] * 256
    speaker_known = _make_speaker(transcription_mod, "S0", embedding=emb_known)
    speaker_unknown = _make_speaker(transcription_mod, "S1", embedding=emb_unknown)
    speaker_no_emb = _make_speaker(transcription_mod, "S2", embedding=None)

    profile = _make_profile(profile_mod, profile_id="p1", name="Alice")
    match = _make_match(profile_mod, profile_id="p1", distance=0.1)

    mock_store = MagicMock()

    def fake_search(embedding, limit=1, distance_threshold=0.4):
        if embedding == emb_known:
            return [match]
        return []

    mock_store.search.side_effect = fake_search
    mock_store.get_profile.return_value = profile

    identifier = matcher_mod.SpeakerIdentifier(
        store=mock_store, match_threshold=0.4, auto_enroll=False
    )
    results = identifier.identify_speakers(
        [speaker_known, speaker_unknown, speaker_no_emb]
    )

    assert len(results) == 3
    assert results[0][1] is profile  # matched
    assert results[1][1] is None  # unknown, no auto_enroll
    assert results[2][1] is None  # no embedding


def test_identify_empty_speaker_list():
    """[] -> []."""
    matcher_mod, transcription_mod, profile_mod = _get_modules()
    mock_store = MagicMock()

    identifier = matcher_mod.SpeakerIdentifier(store=mock_store)
    results = identifier.identify_speakers([])

    assert results == []


def test_enroll_speaker_creates_profile():
    """Verify store.add_profile called with correct data."""
    matcher_mod, transcription_mod, profile_mod = _get_modules()

    embedding = [0.5] * 256
    TimeSegment = transcription_mod.TimeSegment
    speaker = _make_speaker(
        transcription_mod, "S0", embedding=embedding,
        segments=[TimeSegment(start=0.0, end=3.0), TimeSegment(start=5.0, end=8.0)]
    )

    mock_store = MagicMock()
    mock_store.add_profile.side_effect = lambda p: p

    identifier = matcher_mod.SpeakerIdentifier(store=mock_store)
    profile = identifier.enroll_speaker(speaker)

    mock_store.add_profile.assert_called_once()
    assert profile.embedding == embedding
    assert profile.total_duration == 6.0  # 3 + 3


def test_enroll_speaker_without_embedding_raises():
    """ValueError if speaker.embedding is None."""
    import pytest

    matcher_mod, transcription_mod, profile_mod = _get_modules()
    speaker = _make_speaker(transcription_mod, "S0", embedding=None)

    mock_store = MagicMock()
    identifier = matcher_mod.SpeakerIdentifier(store=mock_store)

    with pytest.raises(ValueError, match="embedding is None"):
        identifier.enroll_speaker(speaker)


def test_enroll_speaker_with_name():
    """Name propagated to profile."""
    matcher_mod, transcription_mod, profile_mod = _get_modules()

    embedding = [0.5] * 256
    speaker = _make_speaker(transcription_mod, "S0", embedding=embedding)

    mock_store = MagicMock()
    mock_store.add_profile.side_effect = lambda p: p

    identifier = matcher_mod.SpeakerIdentifier(store=mock_store)
    profile = identifier.enroll_speaker(speaker, name="Bob")

    assert profile.name == "Bob"


def test_update_profile_weighted_average():
    """Known old embedding + new -> verify weighted avg formula."""
    matcher_mod, transcription_mod, profile_mod = _get_modules()

    # Profile has enrollment_count=2, embedding = [1.0, 0.0, 0.0, ...] (256-dim)
    old_emb = [1.0] + [0.0] * 255
    profile = _make_profile(
        profile_mod, profile_id="p1", embedding=old_emb, enrollment_count=2,
        total_duration=10.0
    )

    # Speaker has embedding = [0.0, 1.0, 0.0, ...] (256-dim)
    new_emb = [0.0, 1.0] + [0.0] * 254
    speaker = _make_speaker(transcription_mod, "S0", embedding=new_emb)

    mock_store = MagicMock()
    mock_store.get_profile.return_value = profile
    mock_store.update_profile.side_effect = lambda p: p

    identifier = matcher_mod.SpeakerIdentifier(store=mock_store)
    updated = identifier.update_profile_with_speaker("p1", speaker)

    # Weighted average: (2*1.0 + 1*0.0)/3 = 0.6667, (2*0.0 + 1*1.0)/3 = 0.3333
    # Then L2-normalized
    raw_0 = 2.0 / 3.0
    raw_1 = 1.0 / 3.0
    norm = math.sqrt(raw_0**2 + raw_1**2)
    expected_0 = raw_0 / norm
    expected_1 = raw_1 / norm

    assert abs(updated.embedding[0] - expected_0) < 1e-6
    assert abs(updated.embedding[1] - expected_1) < 1e-6
    for i in range(2, 256):
        assert abs(updated.embedding[i]) < 1e-9


def test_update_profile_increments_enrollment_count():
    """Count goes from N to N+1."""
    matcher_mod, transcription_mod, profile_mod = _get_modules()

    embedding = [0.1] * 256
    profile = _make_profile(
        profile_mod, profile_id="p1", embedding=embedding, enrollment_count=5
    )
    speaker = _make_speaker(transcription_mod, "S0", embedding=embedding)

    mock_store = MagicMock()
    mock_store.get_profile.return_value = profile
    mock_store.update_profile.side_effect = lambda p: p

    identifier = matcher_mod.SpeakerIdentifier(store=mock_store)
    updated = identifier.update_profile_with_speaker("p1", speaker)

    assert updated.enrollment_count == 6


def test_update_profile_normalizes_embedding():
    """L2 norm approx 1.0 after update."""
    matcher_mod, transcription_mod, profile_mod = _get_modules()

    old_emb = [0.5] * 256
    profile = _make_profile(
        profile_mod, profile_id="p1", embedding=old_emb, enrollment_count=3
    )
    new_emb = [0.8] * 256
    speaker = _make_speaker(transcription_mod, "S0", embedding=new_emb)

    mock_store = MagicMock()
    mock_store.get_profile.return_value = profile
    mock_store.update_profile.side_effect = lambda p: p

    identifier = matcher_mod.SpeakerIdentifier(store=mock_store)
    updated = identifier.update_profile_with_speaker("p1", speaker)

    norm = math.sqrt(sum(x * x for x in updated.embedding))
    assert abs(norm - 1.0) < 1e-6


def test_update_profile_not_found_raises():
    """ValueError if profile not found."""
    import pytest

    matcher_mod, transcription_mod, profile_mod = _get_modules()

    embedding = [0.1] * 256
    speaker = _make_speaker(transcription_mod, "S0", embedding=embedding)

    mock_store = MagicMock()
    mock_store.get_profile.return_value = None

    identifier = matcher_mod.SpeakerIdentifier(store=mock_store)

    with pytest.raises(ValueError, match="not found"):
        identifier.update_profile_with_speaker("nonexistent", speaker)


def test_update_profile_no_embedding_raises():
    """ValueError if speaker.embedding is None."""
    import pytest

    matcher_mod, transcription_mod, profile_mod = _get_modules()

    speaker = _make_speaker(transcription_mod, "S0", embedding=None)

    mock_store = MagicMock()

    identifier = matcher_mod.SpeakerIdentifier(store=mock_store)

    with pytest.raises(ValueError, match="embedding is None"):
        identifier.update_profile_with_speaker("p1", speaker)


def test_match_threshold_validation():
    """Threshold < 0 or > 2 -> ValueError."""
    import pytest

    matcher_mod, transcription_mod, profile_mod = _get_modules()
    mock_store = MagicMock()

    with pytest.raises(ValueError, match="match_threshold"):
        matcher_mod.SpeakerIdentifier(store=mock_store, match_threshold=-0.1)

    with pytest.raises(ValueError, match="match_threshold"):
        matcher_mod.SpeakerIdentifier(store=mock_store, match_threshold=2.1)


def test_match_threshold_boundary_exact():
    """Distance exactly == threshold -> included as match."""
    matcher_mod, transcription_mod, profile_mod = _get_modules()

    embedding = [0.1] * 256
    speaker = _make_speaker(transcription_mod, "S0", embedding=embedding)
    profile = _make_profile(profile_mod, profile_id="p1")
    # Distance exactly 0.4 == threshold 0.4
    match = _make_match(profile_mod, profile_id="p1", distance=0.4)

    mock_store = MagicMock()
    mock_store.search.return_value = [match]
    mock_store.get_profile.return_value = profile

    identifier = matcher_mod.SpeakerIdentifier(
        store=mock_store, match_threshold=0.4
    )
    results = identifier.identify_speakers([speaker])

    assert results[0][1] is profile


def test_match_threshold_boundary_over():
    """Distance just above threshold -> excluded (no match returned by store.search)."""
    matcher_mod, transcription_mod, profile_mod = _get_modules()

    embedding = [0.1] * 256
    speaker = _make_speaker(transcription_mod, "S0", embedding=embedding)

    mock_store = MagicMock()
    # Store returns empty because distance > threshold
    mock_store.search.return_value = []

    identifier = matcher_mod.SpeakerIdentifier(
        store=mock_store, match_threshold=0.4, auto_enroll=False
    )
    results = identifier.identify_speakers([speaker])

    assert results[0][1] is None
