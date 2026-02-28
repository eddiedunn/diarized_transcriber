import sys
import types
import importlib
from pathlib import Path
import tempfile
from unittest.mock import MagicMock


def _setup_stubs():
    """Set up common stubs for all engine tests."""
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

    pandas = types.ModuleType("pandas")
    class DataFrame:
        def __init__(self, rows):
            self.data = [dict(row) for row in rows]
        def iterrows(self):
            for idx, row in enumerate(self.data):
                yield idx, row
    pandas.DataFrame = DataFrame
    sys.modules["pandas"] = pandas

    torch = types.ModuleType("torch")
    torch.device = lambda x: x
    sys.modules["torch"] = torch

    numpy = types.ModuleType("numpy")
    numpy.ndarray = list
    numpy.float32 = float
    sys.modules["numpy"] = numpy

    soundfile = types.ModuleType("soundfile")
    soundfile.write = lambda *a, **k: None
    soundfile.read = lambda *a, **k: ([0.0], 16000)
    sys.modules["soundfile"] = soundfile

    pyannote = types.ModuleType("pyannote")
    pyannote_audio = types.ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = type("Pipeline", (), {
        "from_pretrained": classmethod(lambda cls, *a, **k: cls()),
        "to": lambda self, d: self,
    })
    pyannote_audio.Model = type("Model", (), {
        "from_pretrained": classmethod(lambda cls, *a, **k: cls()),
    })
    pyannote_audio.Inference = type("Inference", (), {
        "__init__": lambda self, *a, **k: None,
        "to": lambda self, d: self,
        "crop": lambda self, *a, **k: [0.0] * 256,
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

    pkg = types.ModuleType("diarized_transcriber")
    pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "diarized_transcriber")]
    sys.modules["diarized_transcriber"] = pkg

    return pandas


def _force_reimport():
    """Clear cached module imports to force reimport."""
    sys.modules.pop("diarized_transcriber.transcription.engine", None)
    sys.modules.pop("diarized_transcriber.transcription.audio", None)
    sys.modules.pop("diarized_transcriber.transcription.diarization", None)
    sys.modules.pop("diarized_transcriber.transcription.embeddings", None)
    sys.modules.pop("diarized_transcriber.transcription.gpu", None)
    sys.modules.pop("diarized_transcriber.models.transcription", None)
    sys.modules.pop("diarized_transcriber.models.content", None)
    sys.modules.pop("diarized_transcriber.models.speaker_profile", None)
    sys.modules.pop("diarized_transcriber.exceptions", None)
    sys.modules.pop("diarized_transcriber.types", None)
    sys.modules.pop("diarized_transcriber.identification", None)
    sys.modules.pop("diarized_transcriber.identification.matcher", None)
    sys.modules.pop("diarized_transcriber.storage", None)
    sys.modules.pop("diarized_transcriber.storage.speaker_store", None)


def test_transcription_engine_flow():
    # Stub dependencies
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

    class DataFrame:
        def __init__(self, rows):
            self.data = [dict(row) for row in rows]
        def iterrows(self):
            for idx, row in enumerate(self.data):
                yield idx, row
    pandas = types.ModuleType("pandas")
    pandas.DataFrame = DataFrame
    sys.modules["pandas"] = pandas

    whisperx = types.ModuleType("whisperx")
    class DummyModel:
        def transcribe(self, audio):
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}], "language": "en"}
    def load_model(model_size, device=None, compute_type=None):
        return DummyModel()
    def load_align_model(language_code, device=None):
        return "align_model", {}
    def align(segments, align_model, metadata, audio_array, device, return_char_alignments=False):
        return {"segments": segments, "language": "en"}
    whisperx.load_model = load_model
    whisperx.load_align_model = load_align_model
    whisperx.align = align
    sys.modules["whisperx"] = whisperx

    torch = types.ModuleType("torch")
    torch.device = lambda x: x
    sys.modules["torch"] = torch

    numpy = types.ModuleType("numpy")
    numpy.ndarray = list
    numpy.float32 = float
    sys.modules["numpy"] = numpy

    soundfile = types.ModuleType("soundfile")
    soundfile.write = lambda *a, **k: None
    soundfile.read = lambda *a, **k: ([0.0], 16000)
    sys.modules["soundfile"] = soundfile

    pyannote = types.ModuleType("pyannote")
    pyannote_audio = types.ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = type("Pipeline", (), {"from_pretrained": classmethod(lambda cls, *a, **k: cls()), "to": lambda self, d: self})
    pyannote_audio.Model = type("Model", (), {"from_pretrained": classmethod(lambda cls, *a, **k: cls())})
    pyannote_audio.Inference = type("Inference", (), {
        "__init__": lambda self, *a, **k: None,
        "to": lambda self, d: self,
        "crop": lambda self, *a, **k: [0.0] * 256,
    })
    pyannote.audio = pyannote_audio
    pyannote_core = types.ModuleType("pyannote.core")
    pyannote_core.Segment = type("Segment", (), {"__init__": lambda self, start, end: None})
    pyannote.core = pyannote_core
    sys.modules["pyannote"] = pyannote
    sys.modules["pyannote.audio"] = pyannote_audio
    sys.modules["pyannote.core"] = pyannote_core

    pkg = types.ModuleType("diarized_transcriber")
    pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "diarized_transcriber")]
    sys.modules["diarized_transcriber"] = pkg

    _force_reimport()
    engine_mod = importlib.import_module("diarized_transcriber.transcription.engine")
    content_mod = importlib.import_module("diarized_transcriber.models.content")
    transcription_mod = importlib.import_module("diarized_transcriber.models.transcription")

    engine_mod.verify_gpu_requirements = lambda: "cpu"
    engine_mod.cleanup_gpu_memory = lambda: None
    engine_mod.process_media_content = lambda content: ([0.0], 16000)
    def create_temp_audio_file(data, sr):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        return tmp.name, tmp
    engine_mod.create_temp_audio_file = create_temp_audio_file

    Speaker = transcription_mod.Speaker
    TimeSegment = transcription_mod.TimeSegment
    class DummyDiarization:
        def process_audio(self, audio_file=None, audio_array=None, sample_rate=None, min_speakers=None, max_speakers=None):
            return [Speaker(id="A", segments=[TimeSegment(start=0.0, end=1.0)])]
        def assign_speakers_to_segments(self, speakers, segments):
            for row in segments.data:
                row["speaker"] = speakers[0].id
            return segments
    engine_mod.DiarizationPipeline = lambda device=None: DummyDiarization()

    MediaContent = content_mod.MediaContent
    MediaSource = content_mod.MediaSource

    engine = engine_mod.TranscriptionEngine(model_size="tiny", device="cpu")
    content = MediaContent(
        id="1",
        title="Test",
        media_url="http://example.com/audio.wav",
        source=MediaSource(type="podcast"),
    )

    result = engine.transcribe(content)

    assert result.content_id == "1"
    assert len(result.segments) == 1
    seg = result.segments[0]
    assert seg.text == "hi"
    assert seg.speaker == "A"


def test_default_model_size_is_large_v3_turbo():
    """Verify that TranscriptionEngine defaults to large-v3-turbo."""
    _setup_stubs()

    whisperx = types.ModuleType("whisperx")
    sys.modules["whisperx"] = whisperx

    _force_reimport()
    engine_mod = importlib.import_module("diarized_transcriber.transcription.engine")
    engine_mod.verify_gpu_requirements = lambda: "cpu"

    engine = engine_mod.TranscriptionEngine()
    assert engine.model_size == "large-v3-turbo"


def test_model_size_accepts_all_valid_values():
    """Verify TranscriptionEngine accepts all valid model size values."""
    _setup_stubs()

    whisperx = types.ModuleType("whisperx")
    sys.modules["whisperx"] = whisperx

    _force_reimport()
    engine_mod = importlib.import_module("diarized_transcriber.transcription.engine")
    engine_mod.verify_gpu_requirements = lambda: "cpu"

    valid_sizes = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "large-v3-turbo"]
    for size in valid_sizes:
        engine = engine_mod.TranscriptionEngine(model_size=size, device="cpu")
        assert engine.model_size == size


def test_model_size_literal_includes_large_v3_turbo():
    """Verify the model_size type annotation includes large-v3-turbo."""
    _setup_stubs()

    whisperx = types.ModuleType("whisperx")
    sys.modules["whisperx"] = whisperx

    _force_reimport()
    engine_mod = importlib.import_module("diarized_transcriber.transcription.engine")

    import typing
    hints = typing.get_type_hints(engine_mod.TranscriptionEngine.__init__)
    model_size_hint = hints["model_size"]
    args = typing.get_args(model_size_hint)
    expected = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "large-v3-turbo"}
    assert set(args) == expected


def test_engine_extract_embeddings_default_true():
    """Verify extract_embeddings defaults to True on new engine."""
    _setup_stubs()

    whisperx = types.ModuleType("whisperx")
    sys.modules["whisperx"] = whisperx

    _force_reimport()
    engine_mod = importlib.import_module("diarized_transcriber.transcription.engine")
    engine_mod.verify_gpu_requirements = lambda: "cpu"

    engine = engine_mod.TranscriptionEngine()
    assert engine.extract_embeddings is True


def test_engine_extract_embeddings_false_skips():
    """Set extract_embeddings=False, verify SpeakerEmbeddingExtractor never called."""
    from unittest.mock import MagicMock

    _setup_stubs()

    whisperx = types.ModuleType("whisperx")
    class DummyWhisperModel:
        def transcribe(self, audio):
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}], "language": "en"}
    def load_model(model_size, device=None, compute_type=None):
        return DummyWhisperModel()
    def load_align_model(language_code, device=None):
        return "align_model", {}
    def align(segments, align_model, metadata, audio_array, device, return_char_alignments=False):
        return {"segments": segments, "language": "en"}
    whisperx.load_model = load_model
    whisperx.load_align_model = load_align_model
    whisperx.align = align
    sys.modules["whisperx"] = whisperx

    _force_reimport()
    engine_mod = importlib.import_module("diarized_transcriber.transcription.engine")
    content_mod = importlib.import_module("diarized_transcriber.models.content")
    transcription_mod = importlib.import_module("diarized_transcriber.models.transcription")

    engine_mod.verify_gpu_requirements = lambda: "cpu"
    engine_mod.cleanup_gpu_memory = lambda: None
    engine_mod.process_media_content = lambda content: ([0.0], 16000)
    def create_temp_audio_file(data, sr):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        return tmp.name, tmp
    engine_mod.create_temp_audio_file = create_temp_audio_file

    Speaker = transcription_mod.Speaker
    TimeSegment = transcription_mod.TimeSegment
    class DummyDiarization:
        def process_audio(self, audio_file=None, audio_array=None, sample_rate=None, min_speakers=None, max_speakers=None):
            return [Speaker(id="A", segments=[TimeSegment(start=0.0, end=1.0)])]
        def assign_speakers_to_segments(self, speakers, segments):
            for row in segments.data:
                row["speaker"] = speakers[0].id
            return segments
    engine_mod.DiarizationPipeline = lambda device=None: DummyDiarization()

    # Replace SpeakerEmbeddingExtractor with a mock to verify it is NOT called
    mock_extractor_cls = MagicMock()
    engine_mod.SpeakerEmbeddingExtractor = mock_extractor_cls

    MediaContent = content_mod.MediaContent
    MediaSource = content_mod.MediaSource

    engine = engine_mod.TranscriptionEngine(model_size="tiny", device="cpu", extract_embeddings=False)
    content = MediaContent(
        id="1",
        title="Test",
        media_url="http://example.com/audio.wav",
        source=MediaSource(type="podcast"),
    )
    engine.transcribe(content)

    # SpeakerEmbeddingExtractor should never have been instantiated
    mock_extractor_cls.assert_not_called()
    assert engine._embedding_extractor is None


def test_engine_transcribe_includes_embeddings():
    """Full pipeline mock, verify speakers in result have embeddings after transcribe."""
    from unittest.mock import MagicMock

    _setup_stubs()

    whisperx = types.ModuleType("whisperx")
    class DummyWhisperModel:
        def transcribe(self, audio):
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}], "language": "en"}
    def load_model(model_size, device=None, compute_type=None):
        return DummyWhisperModel()
    def load_align_model(language_code, device=None):
        return "align_model", {}
    def align(segments, align_model, metadata, audio_array, device, return_char_alignments=False):
        return {"segments": segments, "language": "en"}
    whisperx.load_model = load_model
    whisperx.load_align_model = load_align_model
    whisperx.align = align
    sys.modules["whisperx"] = whisperx

    _force_reimport()
    engine_mod = importlib.import_module("diarized_transcriber.transcription.engine")
    content_mod = importlib.import_module("diarized_transcriber.models.content")
    transcription_mod = importlib.import_module("diarized_transcriber.models.transcription")

    engine_mod.verify_gpu_requirements = lambda: "cpu"
    engine_mod.cleanup_gpu_memory = lambda: None
    engine_mod.process_media_content = lambda content: ([0.0], 16000)
    def create_temp_audio_file(data, sr):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        return tmp.name, tmp
    engine_mod.create_temp_audio_file = create_temp_audio_file

    Speaker = transcription_mod.Speaker
    TimeSegment = transcription_mod.TimeSegment
    test_embedding = [0.1] * 256

    class DummyDiarization:
        def process_audio(self, audio_file=None, audio_array=None, sample_rate=None, min_speakers=None, max_speakers=None):
            return [Speaker(id="A", segments=[TimeSegment(start=0.0, end=1.0)])]
        def assign_speakers_to_segments(self, speakers, segments):
            for row in segments.data:
                row["speaker"] = speakers[0].id
            return segments
    engine_mod.DiarizationPipeline = lambda device=None: DummyDiarization()

    # Mock SpeakerEmbeddingExtractor to set embeddings on speakers
    mock_extractor = MagicMock()
    def fake_extract_for_all(audio_file, speakers):
        for s in speakers:
            s.embedding = test_embedding
        return speakers
    mock_extractor.extract_for_all_speakers.side_effect = fake_extract_for_all
    engine_mod.SpeakerEmbeddingExtractor = lambda device=None: mock_extractor

    MediaContent = content_mod.MediaContent
    MediaSource = content_mod.MediaSource

    engine = engine_mod.TranscriptionEngine(model_size="tiny", device="cpu", extract_embeddings=True)
    content = MediaContent(
        id="1",
        title="Test",
        media_url="http://example.com/audio.wav",
        source=MediaSource(type="podcast"),
    )
    result = engine.transcribe(content)

    # Verify the extractor was called
    mock_extractor.extract_for_all_speakers.assert_called_once()

    # Verify speakers have embeddings
    assert result.speakers is not None
    for speaker in result.speakers:
        assert speaker.embedding == test_embedding
        assert len(speaker.embedding) == 256


def test_engine_match_threshold_validation():
    """match_threshold out of range -> ValueError at construction."""
    import pytest

    _setup_stubs()

    whisperx = types.ModuleType("whisperx")
    sys.modules["whisperx"] = whisperx

    _force_reimport()
    engine_mod = importlib.import_module("diarized_transcriber.transcription.engine")
    engine_mod.verify_gpu_requirements = lambda: "cpu"

    with pytest.raises(ValueError, match="match_threshold"):
        engine_mod.TranscriptionEngine(device="cpu", match_threshold=-1.0)

    with pytest.raises(ValueError, match="match_threshold"):
        engine_mod.TranscriptionEngine(device="cpu", match_threshold=2.1)


def test_engine_auto_enroll_without_identify_warns():
    """auto_enroll_speakers=True without identify_speakers logs warning."""
    import logging

    _setup_stubs()

    whisperx = types.ModuleType("whisperx")
    sys.modules["whisperx"] = whisperx

    _force_reimport()
    engine_mod = importlib.import_module("diarized_transcriber.transcription.engine")
    engine_mod.verify_gpu_requirements = lambda: "cpu"

    with MagicMock() as mock_logger:
        from unittest.mock import patch
        with patch.object(engine_mod.logger, "warning") as mock_warn:
            engine_mod.TranscriptionEngine(
                device="cpu",
                auto_enroll_speakers=True,
                identify_speakers=False,
            )
            mock_warn.assert_called_once()
            assert "auto_enroll_speakers" in str(mock_warn.call_args)


def test_engine_identify_speakers_requires_db_path():
    """identify_speakers=True with no db_path -> ValueError."""
    import pytest

    _setup_stubs()

    whisperx = types.ModuleType("whisperx")
    sys.modules["whisperx"] = whisperx

    _force_reimport()
    engine_mod = importlib.import_module("diarized_transcriber.transcription.engine")
    engine_mod.verify_gpu_requirements = lambda: "cpu"

    with pytest.raises(ValueError, match="speaker_db_path"):
        engine_mod.TranscriptionEngine(
            device="cpu", identify_speakers=True, speaker_db_path=None
        )


def _setup_full_pipeline_stubs():
    """Set up stubs for a full transcription pipeline test, returning modules."""
    _setup_stubs()

    whisperx = types.ModuleType("whisperx")

    class DummyWhisperModel:
        def transcribe(self, audio):
            return {
                "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
                "language": "en",
            }

    def load_model(model_size, device=None, compute_type=None):
        return DummyWhisperModel()

    def load_align_model(language_code, device=None):
        return "align_model", {}

    def align(segments, align_model, metadata, audio_array, device, return_char_alignments=False):
        return {"segments": segments, "language": "en"}

    whisperx.load_model = load_model
    whisperx.load_align_model = load_align_model
    whisperx.align = align
    sys.modules["whisperx"] = whisperx

    _force_reimport()
    engine_mod = importlib.import_module("diarized_transcriber.transcription.engine")
    content_mod = importlib.import_module("diarized_transcriber.models.content")
    transcription_mod = importlib.import_module("diarized_transcriber.models.transcription")

    engine_mod.verify_gpu_requirements = lambda: "cpu"
    engine_mod.cleanup_gpu_memory = lambda: None
    engine_mod.process_media_content = lambda content: ([0.0], 16000)

    def create_temp_audio_file(data, sr):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        return tmp.name, tmp

    engine_mod.create_temp_audio_file = create_temp_audio_file

    Speaker = transcription_mod.Speaker
    TimeSegment = transcription_mod.TimeSegment
    test_embedding = [0.1] * 256

    class DummyDiarization:
        def process_audio(self, audio_file=None, audio_array=None, sample_rate=None, min_speakers=None, max_speakers=None):
            return [
                Speaker(id="SPEAKER_00", segments=[TimeSegment(start=0.0, end=5.0)])
            ]

        def assign_speakers_to_segments(self, speakers, segments):
            for row in segments.data:
                row["speaker"] = speakers[0].id
            return segments

    engine_mod.DiarizationPipeline = lambda device=None: DummyDiarization()

    # Mock embedding extractor to set embeddings
    mock_extractor = MagicMock()

    def fake_extract_for_all(audio_file, speakers):
        for s in speakers:
            s.embedding = list(test_embedding)
        return speakers

    mock_extractor.extract_for_all_speakers.side_effect = fake_extract_for_all
    engine_mod.SpeakerEmbeddingExtractor = lambda device=None: mock_extractor

    return engine_mod, content_mod, transcription_mod


def test_engine_identify_speakers_replaces_ids_with_names():
    """Matched speaker's id becomes profile name."""
    engine_mod, content_mod, transcription_mod = _setup_full_pipeline_stubs()

    profile_mod = importlib.import_module(
        "diarized_transcriber.models.speaker_profile"
    )
    mock_profile = profile_mod.SpeakerProfile(
        id="prof-1",
        name="Alice",
        embedding=[0.1] * 256,
        enrollment_count=3,
        total_duration=30.0,
    )
    mock_match = profile_mod.SpeakerMatch(
        profile_id="prof-1", distance=0.1, confidence=0.9
    )

    mock_store = MagicMock()
    mock_store.search.return_value = [mock_match]
    mock_store.get_profile.return_value = mock_profile
    engine_mod.SpeakerProfileStore = lambda path: mock_store

    MediaContent = content_mod.MediaContent
    MediaSource = content_mod.MediaSource

    engine = engine_mod.TranscriptionEngine(
        model_size="tiny",
        device="cpu",
        identify_speakers=True,
        speaker_db_path="/tmp/test_db",
        match_threshold=0.4,
    )
    content = MediaContent(
        id="1",
        title="Test",
        media_url="http://example.com/audio.wav",
        source=MediaSource(type="podcast"),
    )
    result = engine.transcribe(content)

    assert result.speakers is not None
    assert result.speakers[0].id == "Alice"


def test_engine_identify_speakers_preserves_original_id_in_metadata():
    """Original 'SPEAKER_00' stored in metadata."""
    engine_mod, content_mod, transcription_mod = _setup_full_pipeline_stubs()

    profile_mod = importlib.import_module(
        "diarized_transcriber.models.speaker_profile"
    )
    mock_profile = profile_mod.SpeakerProfile(
        id="prof-1",
        name="Alice",
        embedding=[0.1] * 256,
        enrollment_count=3,
        total_duration=30.0,
    )
    mock_match = profile_mod.SpeakerMatch(
        profile_id="prof-1", distance=0.1, confidence=0.9
    )

    mock_store = MagicMock()
    mock_store.search.return_value = [mock_match]
    mock_store.get_profile.return_value = mock_profile
    engine_mod.SpeakerProfileStore = lambda path: mock_store

    MediaContent = content_mod.MediaContent
    MediaSource = content_mod.MediaSource

    engine = engine_mod.TranscriptionEngine(
        model_size="tiny",
        device="cpu",
        identify_speakers=True,
        speaker_db_path="/tmp/test_db",
    )
    content = MediaContent(
        id="1",
        title="Test",
        media_url="http://example.com/audio.wav",
        source=MediaSource(type="podcast"),
    )
    result = engine.transcribe(content)

    assert result.speakers[0].metadata["original_speaker_id"] == "SPEAKER_00"


def test_engine_auto_enroll_creates_profiles():
    """Verify new profile created for unmatched speaker with auto_enroll."""
    engine_mod, content_mod, transcription_mod = _setup_full_pipeline_stubs()

    profile_mod = importlib.import_module(
        "diarized_transcriber.models.speaker_profile"
    )

    mock_store = MagicMock()
    mock_store.search.return_value = []  # no matches
    mock_store.add_profile.side_effect = lambda p: p
    engine_mod.SpeakerProfileStore = lambda path: mock_store

    MediaContent = content_mod.MediaContent
    MediaSource = content_mod.MediaSource

    engine = engine_mod.TranscriptionEngine(
        model_size="tiny",
        device="cpu",
        identify_speakers=True,
        speaker_db_path="/tmp/test_db",
        auto_enroll_speakers=True,
    )
    content = MediaContent(
        id="1",
        title="Test",
        media_url="http://example.com/audio.wav",
        source=MediaSource(type="podcast"),
    )
    engine.transcribe(content)

    # add_profile should have been called for the unmatched speaker
    mock_store.add_profile.assert_called_once()


def test_engine_identify_speakers_false_skips_identification():
    """Default behavior — identify_speakers=False means no identification."""
    engine_mod, content_mod, transcription_mod = _setup_full_pipeline_stubs()

    # SpeakerProfileStore should never be instantiated
    mock_store_cls = MagicMock()
    engine_mod.SpeakerProfileStore = mock_store_cls

    MediaContent = content_mod.MediaContent
    MediaSource = content_mod.MediaSource

    engine = engine_mod.TranscriptionEngine(
        model_size="tiny", device="cpu", identify_speakers=False
    )
    content = MediaContent(
        id="1",
        title="Test",
        media_url="http://example.com/audio.wav",
        source=MediaSource(type="podcast"),
    )
    result = engine.transcribe(content)

    # Store should never have been instantiated
    mock_store_cls.assert_not_called()
    assert engine._speaker_identifier is None
    # Speaker IDs should remain unchanged
    assert result.speakers[0].id == "SPEAKER_00"
