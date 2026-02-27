import sys
import types
import importlib
import math
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# Stub minimal pydantic
pydantic = types.ModuleType("pydantic")


class BaseModel:
    _validators = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._validators = {}

    def __init__(self, **data):
        for field_name, validator_fn in self.__class__._validators.items():
            if field_name in data:
                data[field_name] = validator_fn(data[field_name])

        hints = {}
        for klass in reversed(type(self).__mro__):
            hints.update(getattr(klass, "__annotations__", {}))

        for k, v in data.items():
            setattr(self, k, v)

        for attr_name in hints:
            if attr_name not in data:
                default = getattr(type(self), attr_name, None)
                if isinstance(default, _FieldInfo):
                    if default.default_factory is not None:
                        setattr(self, attr_name, default.default_factory())
                    else:
                        setattr(self, attr_name, default.default)
                elif default is not None:
                    setattr(self, attr_name, default)

    def model_dump(self):
        hints = {}
        for klass in reversed(type(self).__mro__):
            hints.update(getattr(klass, "__annotations__", {}))
        result = {}
        for k in hints:
            if hasattr(self, k):
                val = getattr(self, k)
                if isinstance(val, datetime):
                    result[k] = val.isoformat()
                else:
                    result[k] = val
        return result


class _FieldInfo:
    def __init__(self, default=None, default_factory=None, **kwargs):
        self.default = default
        self.default_factory = default_factory
        self.kwargs = kwargs


def Field(default=None, *, default_factory=None, **kwargs):
    return _FieldInfo(default=default, default_factory=default_factory, **kwargs)


def field_validator(*fields, mode="after"):
    def decorator(fn):
        fn._validate_fields = fields
        fn._validate_mode = mode
        return classmethod(fn) if not isinstance(fn, classmethod) else fn
    return decorator


pydantic.BaseModel = BaseModel
pydantic.Field = Field
pydantic.field_validator = field_validator
sys.modules.setdefault("pydantic", pydantic)

# Stub numpy
numpy = types.ModuleType("numpy")
if "numpy" in sys.modules:
    numpy = sys.modules["numpy"]
numpy.float32 = float
sys.modules["numpy"] = numpy

# Stub pyarrow
pyarrow = types.ModuleType("pyarrow")


def pa_field(name, typ):
    return (name, typ)


def pa_schema(fields):
    return fields


def pa_string():
    return "string"


def pa_int64():
    return "int64"


def pa_float64():
    return "float64"


def pa_float32():
    return "float32"


def pa_list_(typ, size=-1):
    return ("list", typ, size)


pyarrow.field = pa_field
pyarrow.schema = pa_schema
pyarrow.string = pa_string
pyarrow.int64 = pa_int64
pyarrow.float64 = pa_float64
pyarrow.float32 = pa_float32
pyarrow.list_ = pa_list_
sys.modules.setdefault("pyarrow", pyarrow)


# ── In-memory LanceDB mock ──────────────────────────────────────────────
# This mock stores rows in memory and supports the chained query API that
# SpeakerProfileStore uses: table.search().where(...).limit(...).to_list()
# and table.search(vector).metric(...).limit(...).to_list().

class _MockQuery:
    """Mock for LanceDB query builder chain."""

    def __init__(self, rows, vector=None):
        self._rows = list(rows)
        self._vector = vector
        self._where_clause = None
        self._limit_val = 10000
        self._metric = "cosine"

    def where(self, clause):
        self._where_clause = clause
        return self

    def limit(self, n):
        self._limit_val = n
        return self

    def metric(self, m):
        self._metric = m
        return self

    def to_list(self):
        rows = self._rows

        # Apply where filter
        if self._where_clause:
            # Parse simple "id = 'value'" clauses
            if "=" in self._where_clause:
                parts = self._where_clause.split("=", 1)
                field = parts[0].strip()
                value = parts[1].strip().strip("'\"")
                rows = [r for r in rows if str(r.get(field)) == value]

        # Apply vector search with cosine distance
        if self._vector is not None:
            for row in rows:
                emb = row.get("embedding", [])
                row["_distance"] = _cosine_distance(self._vector, emb)
            rows = sorted(rows, key=lambda r: r.get("_distance", 0))

        return rows[:self._limit_val]


class _MockTable:
    """Mock for a LanceDB table with in-memory storage."""

    def __init__(self, name):
        self.name = name
        self._rows = []

    def add(self, rows):
        for row in rows:
            self._rows.append(dict(row))

    def delete(self, where_clause):
        if "=" in where_clause:
            parts = where_clause.split("=", 1)
            field = parts[0].strip()
            value = parts[1].strip().strip("'\"")
            self._rows = [r for r in self._rows if str(r.get(field)) != value]

    def search(self, vector=None):
        return _MockQuery(self._rows, vector=vector)


# Shared state across _MockDB instances at the same path, simulating persistence
_mock_db_registry = {}


class _MockDB:
    """Mock for a LanceDB connection."""

    def __init__(self, path):
        self.path = path
        if path not in _mock_db_registry:
            _mock_db_registry[path] = {}
        self._tables = _mock_db_registry[path]

    def table_names(self):
        return list(self._tables.keys())

    def create_table(self, name, schema=None, data=None):
        self._tables[name] = _MockTable(name)
        return self._tables[name]

    def open_table(self, name):
        if name not in self._tables:
            self._tables[name] = _MockTable(name)
        return self._tables[name]


def _cosine_distance(a, b):
    """Compute cosine distance between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    return 1.0 - similarity


# Stub lancedb module
lancedb_mod = types.ModuleType("lancedb")
lancedb_mod.connect = _MockDB
sys.modules.setdefault("lancedb", lancedb_mod)

# Provide lightweight package stub for diarized_transcriber
pkg = types.ModuleType("diarized_transcriber")
pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "diarized_transcriber")]
sys.modules.setdefault("diarized_transcriber", pkg)

# Import modules under test
exceptions_mod = importlib.import_module("diarized_transcriber.exceptions")
models_transcription = importlib.import_module("diarized_transcriber.models.transcription")
speaker_profile_mod = importlib.import_module("diarized_transcriber.models.speaker_profile")
storage_mod = importlib.import_module("diarized_transcriber.storage.speaker_store")

SpeakerProfile = speaker_profile_mod.SpeakerProfile
SpeakerMatch = speaker_profile_mod.SpeakerMatch
SpeakerProfileStore = storage_mod.SpeakerProfileStore
StorageError = exceptions_mod.StorageError
EMBEDDING_DIM = models_transcription.EMBEDDING_DIM


def _make_store():
    """Create a SpeakerProfileStore with a temp directory."""
    tmp_dir = tempfile.mkdtemp()
    return SpeakerProfileStore(os.path.join(tmp_dir, "test_db"))


def _make_embedding(value=0.0625):
    """Create a 256-dim embedding with uniform value."""
    return [value] * EMBEDDING_DIM


def _make_profile(name=None, embedding=None, **kwargs):
    """Create a SpeakerProfile with defaults."""
    if embedding is None:
        embedding = _make_embedding()
    return SpeakerProfile(name=name, embedding=embedding, **kwargs)


# ── Tests ────────────────────────────────────────────────────────────────


def test_store_creates_db_on_connect():
    """Verify store initializes without error."""
    store = _make_store()
    assert store._table is not None


def test_add_and_retrieve_profile():
    """Add, get by ID, verify all fields match."""
    store = _make_store()
    profile = _make_profile(name="Alice", total_duration=10.0)
    store.add_profile(profile)

    retrieved = store.get_profile(profile.id)
    assert retrieved is not None
    assert retrieved.id == profile.id
    assert retrieved.name == "Alice"
    assert len(retrieved.embedding) == EMBEDDING_DIM
    assert retrieved.enrollment_count == 1


def test_add_duplicate_id_raises():
    """Same profile_id twice raises StorageError."""
    import pytest
    store = _make_store()
    profile = _make_profile(name="Alice")
    store.add_profile(profile)

    # Create another profile with the same ID
    duplicate = SpeakerProfile(
        id=profile.id, embedding=_make_embedding(), name="Bob"
    )
    with pytest.raises(StorageError):
        store.add_profile(duplicate)


def test_get_nonexistent_profile_returns_none():
    """Unknown ID returns None."""
    store = _make_store()
    result = store.get_profile("nonexistent-id")
    assert result is None


def test_search_returns_similar_profiles():
    """Add 2 profiles with known embeddings, query close to one, verify ranking."""
    store = _make_store()

    # Profile A: embedding of all 1.0 (normalized)
    emb_a = [1.0 / math.sqrt(EMBEDDING_DIM)] * EMBEDDING_DIM
    profile_a = _make_profile(name="Alice", embedding=emb_a)
    store.add_profile(profile_a)

    # Profile B: embedding with first half 1.0, second half -1.0 (normalized)
    emb_b = [1.0 / math.sqrt(EMBEDDING_DIM)] * (EMBEDDING_DIM // 2) + \
            [-1.0 / math.sqrt(EMBEDDING_DIM)] * (EMBEDDING_DIM // 2)
    profile_b = _make_profile(name="Bob", embedding=emb_b)
    store.add_profile(profile_b)

    # Query close to A
    query = [1.0 / math.sqrt(EMBEDDING_DIM)] * EMBEDDING_DIM
    matches = store.search(query, distance_threshold=2.0)

    assert len(matches) >= 1
    # Profile A should be the closest match (distance ~0)
    assert matches[0].profile_id == profile_a.id
    assert matches[0].distance < 0.1


def test_search_respects_distance_threshold():
    """Far embedding returns no matches with tight threshold."""
    store = _make_store()

    emb = [1.0 / math.sqrt(EMBEDDING_DIM)] * EMBEDDING_DIM
    profile = _make_profile(embedding=emb)
    store.add_profile(profile)

    # Query with opposite embedding
    query = [-1.0 / math.sqrt(EMBEDDING_DIM)] * EMBEDDING_DIM
    matches = store.search(query, distance_threshold=0.01)
    assert len(matches) == 0


def test_search_respects_limit():
    """Add 10 profiles, limit=3, verify max 3 returned."""
    store = _make_store()

    for i in range(10):
        emb = _make_embedding(value=float(i + 1) / 100)
        store.add_profile(_make_profile(embedding=emb))

    query = _make_embedding(value=0.01)
    matches = store.search(query, limit=3, distance_threshold=2.0)
    assert len(matches) <= 3


def test_search_empty_store_returns_empty():
    """No profiles returns empty list."""
    store = _make_store()
    matches = store.search(_make_embedding(), distance_threshold=2.0)
    assert matches == []


def test_search_validates_embedding_dimension():
    """Wrong dim raises ValueError."""
    import pytest
    store = _make_store()
    with pytest.raises(ValueError):
        store.search([0.1] * 128)


def test_delete_profile():
    """Add, delete, verify get returns None."""
    store = _make_store()
    profile = _make_profile(name="Alice")
    store.add_profile(profile)

    store.delete_profile(profile.id)
    assert store.get_profile(profile.id) is None


def test_delete_nonexistent_is_idempotent():
    """No error on missing ID."""
    store = _make_store()
    # Should not raise
    store.delete_profile("nonexistent-id")


def test_list_profiles_empty():
    """New store returns empty list."""
    store = _make_store()
    profiles = store.list_profiles()
    assert profiles == []


def test_list_profiles_returns_all():
    """Add 3, list returns 3."""
    store = _make_store()
    for i in range(3):
        store.add_profile(_make_profile(name=f"Speaker{i}"))

    profiles = store.list_profiles()
    assert len(profiles) == 3


def test_merge_profiles_weighted_average():
    """2 profiles with known embeddings, verify merged embedding is correct weighted average."""
    store = _make_store()

    # Source: enrollment_count=2, embedding=[0.1]*256
    emb_src = [0.1] * EMBEDDING_DIM
    source = _make_profile(name="Source", embedding=emb_src, enrollment_count=2)
    store.add_profile(source)

    # Target: enrollment_count=3, embedding=[0.2]*256
    emb_tgt = [0.2] * EMBEDDING_DIM
    target = _make_profile(name="Target", embedding=emb_tgt, enrollment_count=3)
    store.add_profile(target)

    merged = store.merge_profiles(source.id, target.id)

    # Weighted average: (0.1*2 + 0.2*3) / 5 = 0.16 for each dim
    # Then L2-normalized: 0.16 / (0.16 * sqrt(256)) = 1/16 = 0.0625
    expected_raw = 0.16
    expected_norm = math.sqrt(sum(expected_raw ** 2 for _ in range(EMBEDDING_DIM)))
    expected_val = expected_raw / expected_norm
    assert abs(merged.embedding[0] - expected_val) < 1e-6


def test_merge_profiles_source_deleted():
    """After merge, source profile is gone."""
    store = _make_store()
    source = _make_profile(name="Source")
    target = _make_profile(name="Target")
    store.add_profile(source)
    store.add_profile(target)

    store.merge_profiles(source.id, target.id)
    assert store.get_profile(source.id) is None


def test_merge_profiles_target_updated():
    """enrollment_count summed, updated_at refreshed."""
    store = _make_store()
    source = _make_profile(name="Source", enrollment_count=2)
    target = _make_profile(name="Target", enrollment_count=3)
    store.add_profile(source)
    store.add_profile(target)

    before_merge = datetime.now(timezone.utc)
    merged = store.merge_profiles(source.id, target.id)
    assert merged.enrollment_count == 5


def test_merge_profiles_result_normalized():
    """Merged embedding L2 norm approximately 1.0."""
    store = _make_store()
    source = _make_profile(embedding=[0.3] * EMBEDDING_DIM, enrollment_count=1)
    target = _make_profile(embedding=[0.7] * EMBEDDING_DIM, enrollment_count=1)
    store.add_profile(source)
    store.add_profile(target)

    merged = store.merge_profiles(source.id, target.id)
    norm = math.sqrt(sum(x * x for x in merged.embedding))
    assert abs(norm - 1.0) < 1e-6


def test_merge_same_profile_raises():
    """source_id == target_id raises ValueError."""
    import pytest
    store = _make_store()
    profile = _make_profile()
    store.add_profile(profile)

    with pytest.raises(ValueError):
        store.merge_profiles(profile.id, profile.id)


def test_merge_nonexistent_source_raises():
    """Missing source raises StorageError."""
    import pytest
    store = _make_store()
    target = _make_profile()
    store.add_profile(target)

    with pytest.raises(StorageError):
        store.merge_profiles("nonexistent", target.id)


def test_merge_nonexistent_target_raises():
    """Missing target raises StorageError."""
    import pytest
    store = _make_store()
    source = _make_profile()
    store.add_profile(source)

    with pytest.raises(StorageError):
        store.merge_profiles(source.id, "nonexistent")


def test_update_profile():
    """Modify name, update, verify persisted."""
    store = _make_store()
    profile = _make_profile(name="OldName")
    store.add_profile(profile)

    profile.name = "NewName"
    store.update_profile(profile)

    retrieved = store.get_profile(profile.id)
    assert retrieved is not None
    assert retrieved.name == "NewName"


def test_store_persistence_across_reconnect():
    """Add profile, create new store instance with same path, verify profile still there."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "persist_db")

    store1 = SpeakerProfileStore(db_path)
    profile = _make_profile(name="Persistent")
    store1.add_profile(profile)

    # Create new store pointing to same path
    store2 = SpeakerProfileStore(db_path)
    retrieved = store2.get_profile(profile.id)
    assert retrieved is not None
    assert retrieved.name == "Persistent"
