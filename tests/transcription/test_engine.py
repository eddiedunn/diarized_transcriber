import sys
import types
import importlib
from pathlib import Path
import tempfile


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
    pydantic.BaseModel = BaseModel
    pydantic.Field = Field
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

    pkg = types.ModuleType("diarized_transcriber")
    pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "diarized_transcriber")]
    sys.modules["diarized_transcriber"] = pkg

    engine_mod = importlib.import_module("diarized_transcriber.transcription.engine")
    content_mod = importlib.import_module("diarized_transcriber.models.content")
    transcription_mod = importlib.import_module("diarized_transcriber.models.transcription")

    engine_mod.verify_gpu_requirements = lambda: "cpu"
    engine_mod.cleanup_gpu_memory = lambda: None

    audio_mod = importlib.import_module("diarized_transcriber.transcription.audio")
    audio_mod.process_media_content = lambda content: ([0.0], 16000)
    def create_temp_audio_file(data, sr):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        return tmp.name, tmp
    audio_mod.create_temp_audio_file = create_temp_audio_file

    Speaker = transcription_mod.Speaker
    TimeSegment = transcription_mod.TimeSegment
    class DummyDiarization:
        def process_audio(self, path, min_speakers=None, max_speakers=None):
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

