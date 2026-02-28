"""Speaker profile management endpoints."""

import os
from typing import Optional

from fastapi import APIRouter, HTTPException

from ..exceptions import StorageError
from ..models.speaker_profile import SpeakerProfile
from ..storage.speaker_store import SpeakerProfileStore

from .speaker_schemas import (
    EnrollSpeakerRequest,
    MergeProfilesRequest,
    NameSpeakerRequest,
    SearchSpeakersRequest,
    SpeakerProfileListResponse,
    SpeakerProfileResponse,
    SpeakerSearchResponse,
)

router = APIRouter(prefix="/speakers", tags=["speakers"])

_store: Optional[SpeakerProfileStore] = None


def get_store() -> SpeakerProfileStore:
    """Lazily initialize and return the SpeakerProfileStore singleton."""
    global _store
    if _store is None:
        db_path = os.environ.get("SPEAKER_DB_PATH", "./speaker_profiles_db")
        _store = SpeakerProfileStore(db_path=db_path)
    return _store


def _profile_to_response(profile: SpeakerProfile) -> SpeakerProfileResponse:
    """Convert a SpeakerProfile model to a SpeakerProfileResponse."""
    return SpeakerProfileResponse(
        id=profile.id,
        name=profile.name,
        embedding=profile.embedding,
        enrollment_count=profile.enrollment_count,
        total_duration=profile.total_duration,
        created_at=profile.created_at.isoformat(),
        updated_at=profile.updated_at.isoformat(),
        metadata=profile.metadata,
    )


@router.get("/", response_model=SpeakerProfileListResponse)
def list_profiles() -> SpeakerProfileListResponse:
    """List all speaker profiles."""
    store = get_store()
    profiles = store.list_profiles()
    items = [_profile_to_response(p) for p in profiles]
    return SpeakerProfileListResponse(profiles=items, count=len(items))


@router.get("/{profile_id}", response_model=SpeakerProfileResponse)
def get_profile(profile_id: str) -> SpeakerProfileResponse:
    """Get a single speaker profile by ID."""
    store = get_store()
    profile = store.get_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    return _profile_to_response(profile)


@router.post("/enroll", response_model=SpeakerProfileResponse, status_code=200)
def enroll_speaker(req: EnrollSpeakerRequest) -> SpeakerProfileResponse:
    """Create a new speaker profile from an embedding."""
    store = get_store()
    profile = SpeakerProfile(
        name=req.name,
        embedding=req.embedding,
    )
    try:
        created = store.add_profile(profile)
    except StorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _profile_to_response(created)


@router.put("/{profile_id}/name", response_model=SpeakerProfileResponse)
def name_speaker(
    profile_id: str, req: NameSpeakerRequest
) -> SpeakerProfileResponse:
    """Rename a speaker profile."""
    store = get_store()
    profile = store.get_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    profile.name = req.name
    try:
        updated = store.update_profile(profile)
    except StorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _profile_to_response(updated)


@router.delete("/{profile_id}", status_code=200)
def delete_profile(profile_id: str) -> dict:
    """Delete a speaker profile."""
    store = get_store()
    profile = store.get_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    store.delete_profile(profile_id)
    return {"detail": "Profile deleted"}


@router.post("/search", response_model=SpeakerSearchResponse)
def search_speakers(req: SearchSpeakersRequest) -> SpeakerSearchResponse:
    """Search for similar speaker profiles by embedding."""
    store = get_store()
    try:
        matches = store.search(
            embedding=req.embedding,
            limit=req.limit,
            distance_threshold=req.threshold,
        )
    except (ValueError, StorageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result_matches = []
    for match in matches:
        profile = store.get_profile(match.profile_id)
        entry = {
            "profile_id": match.profile_id,
            "distance": match.distance,
            "confidence": match.confidence,
            "profile": _profile_to_response(profile).model_dump()
            if profile
            else None,
        }
        result_matches.append(entry)

    return SpeakerSearchResponse(
        matches=result_matches, count=len(result_matches)
    )


@router.post("/merge", response_model=SpeakerProfileResponse)
def merge_profiles(req: MergeProfilesRequest) -> SpeakerProfileResponse:
    """Merge two speaker profiles."""
    store = get_store()
    try:
        merged = store.merge_profiles(
            source_id=req.source_profile_id,
            target_id=req.target_profile_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except StorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _profile_to_response(merged)
