"""Lightweight speaker fingerprinting endpoint.

Extracts voice embeddings from audio at specified timestamps without
running full transcription or creating any speaker profiles.
"""

import json
import logging
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, UploadFile

from ..config import get_settings
from ..models.transcription import Speaker, TimeSegment
from ..storage.speaker_store import SpeakerProfileStore
from ..transcription.embeddings import SpeakerEmbeddingExtractor

logger = logging.getLogger(__name__)

router = APIRouter(tags=["fingerprint"])

_extractor: Optional[SpeakerEmbeddingExtractor] = None
_store: Optional[SpeakerProfileStore] = None


def _get_extractor() -> SpeakerEmbeddingExtractor:
    global _extractor
    if _extractor is None:
        _extractor = SpeakerEmbeddingExtractor(device="cuda")
    return _extractor


def _get_store() -> SpeakerProfileStore:
    global _store
    if _store is None:
        db_path = get_settings().speaker_db_path
        _store = SpeakerProfileStore(db_path=db_path)
    return _store


def _build_speakers_from_input(
    segments: Optional[str], speakers: Optional[str]
) -> list[Speaker]:
    """Parse either flat segments or grouped speakers input into Speaker objects.

    Accepts two formats:
      speakers: [{"id": "SPEAKER_00", "segments": [{"start": 0, "end": 5}, ...]}, ...]
        → one Speaker per entry, multiple time segments each (better embeddings)

      segments: [{"start": 0, "end": 5}, {"start": 10, "end": 15}]
        → one Speaker per segment (backward compatible)
    """
    if speakers:
        try:
            speaker_list = json.loads(speakers)
            if not isinstance(speaker_list, list) or not speaker_list:
                raise ValueError("speakers must be a non-empty JSON array")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid speakers: {e}")

        result = []
        for i, spk in enumerate(speaker_list):
            spk_id = spk.get("id", f"speaker_{i}")
            spk_segs = spk.get("segments", [])
            if not spk_segs:
                raise HTTPException(
                    status_code=400,
                    detail=f"Speaker {spk_id} has no segments",
                )
            time_segments = []
            for j, seg in enumerate(spk_segs):
                if "start" not in seg or "end" not in seg:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Speaker {spk_id} segment {j} missing 'start' or 'end'",
                    )
                if seg["end"] <= seg["start"]:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Speaker {spk_id} segment {j}: end must be > start",
                    )
                time_segments.append(TimeSegment(start=seg["start"], end=seg["end"]))
            result.append(Speaker(id=spk_id, segments=time_segments))
        return result

    if segments:
        try:
            segment_list = json.loads(segments)
            if not isinstance(segment_list, list) or not segment_list:
                raise ValueError("segments must be a non-empty JSON array")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid segments: {e}")

        result = []
        for i, seg in enumerate(segment_list):
            if "start" not in seg or "end" not in seg:
                raise HTTPException(
                    status_code=400,
                    detail=f"Segment {i} missing 'start' or 'end'",
                )
            if seg["end"] <= seg["start"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Segment {i}: end must be > start",
                )
            result.append(Speaker(
                id=f"fingerprint_{i}",
                segments=[TimeSegment(start=seg["start"], end=seg["end"])],
            ))
        return result

    raise HTTPException(
        status_code=400,
        detail="Either 'speakers' or 'segments' parameter is required",
    )


@router.post("/v1/fingerprint")
async def fingerprint(
    audio: UploadFile,
    segments: Optional[str] = Form(
        None,
        description='JSON array of {"start": float, "end": float} — one embedding per entry',
    ),
    speakers: Optional[str] = Form(
        None,
        description='JSON array of {"id": str, "segments": [{"start", "end"}, ...]} '
        "— groups segments per speaker for better embeddings",
    ),
    match_profiles: bool = Form(True),
    threshold: float = Form(0.4),
) -> dict:
    """Extract speaker embeddings from audio at specified timestamps.

    Two input modes:
      - **speakers** (recommended): Group segments by speaker label. Each speaker
        gets one embedding averaged across all their segments — more audio means
        more accurate fingerprints.
      - **segments**: Flat list of time ranges. Each becomes an independent embedding.

    Unlike /v1/transcribe, this does NOT run speech-to-text, diarization,
    or auto-enrollment. It only extracts voice fingerprints and optionally
    matches them against stored profiles.
    """
    speaker_inputs = _build_speakers_from_input(segments, speakers)

    tmp_path = None
    try:
        suffix = os.path.splitext(audio.filename or "upload.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            contents = await audio.read()
            tmp.write(contents)

        extractor = _get_extractor()
        store = _get_store() if match_profiles else None

        results = []
        for speaker in speaker_inputs:
            total_duration = sum(s.end - s.start for s in speaker.segments)

            try:
                embedding = extractor.extract_for_speaker(tmp_path, speaker)
            except Exception as e:
                logger.warning(
                    "Embedding extraction failed for %s (%.1fs across %d segments): %s",
                    speaker.id, total_duration, len(speaker.segments), e,
                )
                results.append({
                    "speaker_id": speaker.id,
                    "embedding": None,
                    "match": None,
                    "total_duration": total_duration,
                    "segment_count": len(speaker.segments),
                    "error": str(e),
                })
                continue

            # Search for matching profile
            match_data = None
            if store is not None:
                matches = store.search(
                    embedding=embedding,
                    limit=1,
                    distance_threshold=threshold,
                )
                if matches:
                    profile = store.get_profile(matches[0].profile_id)
                    if profile:
                        match_data = {
                            "profile_id": profile.id,
                            "name": profile.name,
                            "confidence": matches[0].confidence,
                        }

            results.append({
                "speaker_id": speaker.id,
                "embedding": embedding,
                "match": match_data,
                "total_duration": total_duration,
                "segment_count": len(speaker.segments),
            })

        return {"embeddings": results}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("fingerprint failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
