"""Cross-recording speaker identification using stored profiles."""

import logging
import math
from typing import List, Optional, Tuple

from ..models.speaker_profile import SpeakerMatch, SpeakerProfile
from ..models.transcription import Speaker
from ..storage.speaker_store import SpeakerProfileStore

logger = logging.getLogger(__name__)


class SpeakerIdentifier:
    """Identifies speakers across recordings using embedding similarity."""

    def __init__(
        self,
        store: SpeakerProfileStore,
        match_threshold: float = 0.4,
        auto_enroll: bool = False,
    ):
        if match_threshold < 0.0 or match_threshold > 2.0:
            raise ValueError(
                f"match_threshold must be between 0.0 and 2.0, got {match_threshold}"
            )
        self.store = store
        self.match_threshold = match_threshold
        self.auto_enroll = auto_enroll

    def identify_speakers(
        self, speakers: List[Speaker]
    ) -> List[Tuple[Speaker, Optional[SpeakerProfile]]]:
        """Identify a list of speakers against stored profiles.

        For each speaker:
        - If embedding is None, skip with warning and return (speaker, None).
        - If a match is found above threshold, return (speaker, matched_profile).
        - If no match and auto_enroll is True, create a new profile.
        - If no match and auto_enroll is False, return (speaker, None).

        Args:
            speakers: List of Speaker objects with optional embeddings.

        Returns:
            List of (speaker, profile_or_none) tuples.
        """
        results: List[Tuple[Speaker, Optional[SpeakerProfile]]] = []

        for speaker in speakers:
            if speaker.embedding is None:
                logger.warning(
                    "Speaker %s has no embedding, skipping identification",
                    speaker.id,
                )
                results.append((speaker, None))
                continue

            matches = self.store.search(
                embedding=speaker.embedding,
                limit=1,
                distance_threshold=self.match_threshold,
            )

            if matches:
                best = matches[0]
                profile = self.store.get_profile(best.profile_id)
                if profile is not None:
                    logger.info(
                        "Speaker %s matched profile %s (distance=%.4f)",
                        speaker.id,
                        profile.id,
                        best.distance,
                    )
                    results.append((speaker, profile))
                    continue

            if self.auto_enroll:
                profile = self.enroll_speaker(speaker)
                logger.info(
                    "Auto-enrolled speaker %s as profile %s",
                    speaker.id,
                    profile.id,
                )
                results.append((speaker, profile))
            else:
                results.append((speaker, None))

        return results

    def enroll_speaker(
        self, speaker: Speaker, name: Optional[str] = None
    ) -> SpeakerProfile:
        """Create a new profile from a speaker's embedding.

        Args:
            speaker: Speaker with a non-None embedding.
            name: Optional display name for the profile.

        Returns:
            The newly created SpeakerProfile.

        Raises:
            ValueError: If speaker.embedding is None.
        """
        if speaker.embedding is None:
            raise ValueError(
                f"Cannot enroll speaker {speaker.id}: embedding is None"
            )

        total_duration = sum(
            seg.end - seg.start for seg in speaker.segments
        )

        profile = SpeakerProfile(
            name=name,
            embedding=list(speaker.embedding),
            total_duration=total_duration,
        )
        self.store.add_profile(profile)
        return profile

    def update_profile_with_speaker(
        self, profile_id: str, speaker: Speaker
    ) -> SpeakerProfile:
        """Update an existing profile's embedding with a new speaker observation.

        Uses weighted average based on enrollment_count, then L2-normalizes.

        Args:
            profile_id: ID of the profile to update.
            speaker: Speaker with a non-None embedding.

        Returns:
            The updated SpeakerProfile.

        Raises:
            ValueError: If speaker.embedding is None or profile not found.
        """
        if speaker.embedding is None:
            raise ValueError(
                f"Cannot update profile with speaker {speaker.id}: "
                "embedding is None"
            )

        profile = self.store.get_profile(profile_id)
        if profile is None:
            raise ValueError(f"Profile {profile_id} not found")

        old_weight = profile.enrollment_count
        new_weight = 1
        total_weight = old_weight + new_weight

        merged = [
            (o * old_weight + n * new_weight) / total_weight
            for o, n in zip(profile.embedding, speaker.embedding)
        ]

        norm = math.sqrt(sum(x * x for x in merged))
        if norm > 0:
            merged = [x / norm for x in merged]

        profile.embedding = merged
        profile.enrollment_count = total_weight

        if profile.total_duration is not None:
            speaker_duration = sum(
                seg.end - seg.start for seg in speaker.segments
            )
            profile.total_duration += speaker_duration

        return self.store.update_profile(profile)
