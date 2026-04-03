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


@router.post("/v1/fingerprint")
async def fingerprint(
    audio: UploadFile,
    segments: str = Form(..., description='JSON array of {"start": float, "end": float} time ranges'),
    match_profiles: bool = Form(True),
    threshold: float = Form(0.4),
) -> dict:
    """Extract speaker embeddings from audio at specified timestamps.

    Unlike /v1/transcribe, this does NOT run speech-to-text, diarization,
    or auto-enrollment. It only extracts voice fingerprints and optionally
    matches them against stored profiles.
    """
    # Parse segments JSON
    try:
        segment_list = json.loads(segments)
        if not isinstance(segment_list, list) or not segment_list:
            raise ValueError("segments must be a non-empty JSON array")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid segments: {e}")

    # Validate each segment has start/end
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

    tmp_path = None
    try:
        # Save uploaded audio to temp file
        suffix = os.path.splitext(audio.filename or "upload.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            contents = await audio.read()
            tmp.write(contents)

        extractor = _get_extractor()
        store = _get_store() if match_profiles else None

        results = []
        for i, seg in enumerate(segment_list):
            # Build a Speaker object with the given time segment
            speaker = Speaker(
                id=f"fingerprint_{i}",
                segments=[TimeSegment(start=seg["start"], end=seg["end"])],
            )

            try:
                embedding = extractor.extract_for_speaker(tmp_path, speaker)
            except Exception as e:
                logger.warning("Embedding extraction failed for segment %d: %s", i, e)
                results.append({
                    "segment_index": i,
                    "embedding": None,
                    "match": None,
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
                "segment_index": i,
                "embedding": embedding,
                "match": match_data,
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
