"""Speaker profile storage backed by LanceDB."""

import logging
import math
from datetime import datetime, timezone
from typing import List, Optional

import lancedb
import numpy as np
import pyarrow as pa

from ..exceptions import StorageError
from ..models.speaker_profile import SpeakerMatch, SpeakerProfile
from ..models.transcription import EMBEDDING_DIM

logger = logging.getLogger(__name__)

TABLE_NAME = "speaker_profiles"


class SpeakerProfileStore:
    """Persistent speaker profile storage using LanceDB for vector search."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db = lancedb.connect(db_path)
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the profiles table if it doesn't exist."""
        if TABLE_NAME not in self._db.table_names():
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("name", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), EMBEDDING_DIM)),
                pa.field("enrollment_count", pa.int64()),
                pa.field("total_duration", pa.float64()),
                pa.field("created_at", pa.string()),
                pa.field("updated_at", pa.string()),
                pa.field("metadata_json", pa.string()),
            ])
            self._db.create_table(TABLE_NAME, schema=schema)
        self._table = self._db.open_table(TABLE_NAME)

    def _profile_to_row(self, profile: SpeakerProfile) -> dict:
        """Convert a SpeakerProfile to a LanceDB row dict."""
        import json
        return {
            "id": profile.id,
            "name": profile.name or "",
            "embedding": profile.embedding,
            "enrollment_count": profile.enrollment_count,
            "total_duration": profile.total_duration if profile.total_duration is not None else -1.0,
            "created_at": profile.created_at.isoformat(),
            "updated_at": profile.updated_at.isoformat(),
            "metadata_json": json.dumps(profile.metadata),
        }

    def _row_to_profile(self, row: dict) -> SpeakerProfile:
        """Convert a LanceDB row dict to a SpeakerProfile."""
        import json
        name = row.get("name")
        if name == "":
            name = None
        total_duration = row.get("total_duration")
        if total_duration is not None and total_duration < 0:
            total_duration = None
        return SpeakerProfile(
            id=row["id"],
            name=name,
            embedding=list(row["embedding"]),
            enrollment_count=row["enrollment_count"],
            total_duration=total_duration,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row.get("metadata_json", "{}")),
        )

    def add_profile(self, profile: SpeakerProfile) -> SpeakerProfile:
        """Add a new speaker profile to the store.

        Raises:
            StorageError: If a profile with the same ID already exists.
        """
        existing = self.get_profile(profile.id)
        if existing is not None:
            raise StorageError(
                f"Profile with ID {profile.id} already exists"
            )
        self._table.add([self._profile_to_row(profile)])
        return profile

    def get_profile(self, profile_id: str) -> Optional[SpeakerProfile]:
        """Get a profile by ID, or None if not found."""
        results = (
            self._table.search()
            .where(f"id = '{profile_id}'")
            .limit(1)
            .to_list()
        )
        if not results:
            return None
        return self._row_to_profile(results[0])

    def list_profiles(self) -> List[SpeakerProfile]:
        """Return all stored profiles."""
        rows = self._table.search().limit(10000).to_list()
        return [self._row_to_profile(row) for row in rows]

    def delete_profile(self, profile_id: str) -> None:
        """Delete a profile by ID. No-op if not found (idempotent)."""
        self._table.delete(f"id = '{profile_id}'")

    def update_profile(self, profile: SpeakerProfile) -> SpeakerProfile:
        """Update an existing profile (delete + re-add).

        Raises:
            StorageError: If the profile doesn't exist.
        """
        existing = self.get_profile(profile.id)
        if existing is None:
            raise StorageError(
                f"Profile with ID {profile.id} not found"
            )
        profile.updated_at = datetime.now(timezone.utc)
        self._table.delete(f"id = '{profile.id}'")
        self._table.add([self._profile_to_row(profile)])
        return profile

    def search(
        self,
        embedding: List[float],
        limit: int = 10,
        distance_threshold: float = 0.4,
    ) -> List[SpeakerMatch]:
        """Search for similar speaker profiles by embedding.

        Args:
            embedding: Query embedding (256-dim).
            limit: Max results to return (1-100).
            distance_threshold: Max cosine distance for matches (0.0-2.0).

        Returns:
            List of SpeakerMatch sorted by distance (closest first).

        Raises:
            ValueError: If embedding dimension is wrong or params out of range.
        """
        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding must be {EMBEDDING_DIM}-dimensional, got {len(embedding)}"
            )
        if limit < 1 or limit > 100:
            raise ValueError(f"limit must be between 1 and 100, got {limit}")
        if distance_threshold < 0.0 or distance_threshold > 2.0:
            raise ValueError(
                f"distance_threshold must be between 0.0 and 2.0, got {distance_threshold}"
            )

        results = (
            self._table
            .search(embedding)
            .metric("cosine")
            .limit(limit)
            .to_list()
        )

        matches = []
        for row in results:
            distance = row.get("_distance", 0.0)
            if distance <= distance_threshold:
                matches.append(
                    SpeakerMatch(
                        profile_id=row["id"],
                        distance=distance,
                        confidence=1.0 - distance,
                    )
                )
        return matches

    def merge_profiles(
        self, source_id: str, target_id: str
    ) -> SpeakerProfile:
        """Merge source profile into target using weighted average embedding.

        The source profile is deleted after merging. The target profile's
        enrollment_count is summed and embedding is a weighted average,
        then re-normalized to L2=1.0.

        Args:
            source_id: Profile ID to merge from (will be deleted).
            target_id: Profile ID to merge into (will be updated).

        Returns:
            Updated target profile.

        Raises:
            ValueError: If source_id == target_id.
            StorageError: If either profile doesn't exist.
        """
        if source_id == target_id:
            raise ValueError("Cannot merge a profile into itself")

        source = self.get_profile(source_id)
        if source is None:
            raise StorageError(f"Source profile {source_id} not found")

        target = self.get_profile(target_id)
        if target is None:
            raise StorageError(f"Target profile {target_id} not found")

        # Weighted average of embeddings
        src_weight = source.enrollment_count
        tgt_weight = target.enrollment_count
        total_weight = src_weight + tgt_weight

        merged_embedding = [
            (s * src_weight + t * tgt_weight) / total_weight
            for s, t in zip(source.embedding, target.embedding)
        ]

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in merged_embedding))
        if norm > 0:
            merged_embedding = [x / norm for x in merged_embedding]

        target.embedding = merged_embedding
        target.enrollment_count = total_weight

        # Sum durations if both are set
        if source.total_duration is not None and target.total_duration is not None:
            target.total_duration = source.total_duration + target.total_duration

        # Delete source, update target
        self.delete_profile(source_id)
        self._table.delete(f"id = '{target_id}'")
        target.updated_at = datetime.now(timezone.utc)
        self._table.add([self._profile_to_row(target)])

        return target
