import json
import os
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

# --- PATH CONFIGURATION ---
# Cloud Run uses /tmp because it's the only writable directory.
# The generator stores its local output beside this module. The legacy
# current-working-directory path remains a fallback for existing setups.
CLOUD_PATH = Path("/tmp/embeddings.sqlite")
LOCAL_PATH = Path(__file__).with_name("embeddings.sqlite")
LEGACY_LOCAL_PATH = Path("./embeddings.sqlite")
# --------------------------

EmbeddingRow = tuple[int, np.ndarray]

_cached_vectors: list[EmbeddingRow] | None = None
_cached_path: Path | None = None


def _as_vector(item_id: int, raw_vector: object) -> np.ndarray:
    try:
        vector = np.asarray(raw_vector, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid cached embedding for grocery item {item_id}"
        ) from exc

    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"Invalid cached embedding for grocery item {item_id}")
    return vector


def _existing_database_uri(path: str | Path, *, mode: str) -> tuple[Path, str]:
    db_path = Path(path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Embeddings database not found at {db_path}")
    return db_path, f"{db_path.resolve().as_uri()}?mode={mode}"


def _clear_memory_cache() -> None:
    global _cached_path, _cached_vectors

    _cached_vectors = None
    _cached_path = None


def get_embeddings_db_path() -> Path:
    """Resolve the cache path without creating an empty SQLite database."""

    configured_path = os.getenv("EMBEDDINGS_DB_PATH", "").strip()
    if configured_path:
        return Path(configured_path).expanduser()

    for candidate in (CLOUD_PATH, LOCAL_PATH, LEGACY_LOCAL_PATH):
        if candidate.is_file():
            return candidate
    return LOCAL_PATH


def read_cached_embeddings(path: str | Path) -> list[EmbeddingRow]:
    """Read and validate cached vectors, raising on missing or corrupt data."""

    db_path, read_only_uri = _existing_database_uri(path, mode="ro")

    try:
        with sqlite3.connect(read_only_uri, uri=True) as connection:
            rows = connection.execute(
                "SELECT grocery_item_id, embedding "
                "FROM grocery_item_embeddings"
            ).fetchall()
    except sqlite3.Error as exc:
        raise RuntimeError(
            f"Could not read embeddings database at {db_path}: {exc}"
        ) from exc

    vectors: list[EmbeddingRow] = []
    for item_id, raw_embedding in rows:
        try:
            decoded_embedding = json.loads(raw_embedding)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                f"Invalid cached embedding for grocery item {item_id}"
            ) from exc
        vector = _as_vector(int(item_id), decoded_embedding)
        vectors.append((int(item_id), vector))
    return vectors


def get_cached_embedding_dimension(path: str | Path) -> int | None:
    """Return the cache dimension and reject a mixed-dimension cache."""

    dimensions = {
        vector.size for _, vector in read_cached_embeddings(path)
    }
    if len(dimensions) > 1:
        raise ValueError(
            "Cached embeddings have inconsistent dimensions: "
            f"{sorted(dimensions)}"
        )
    return next(iter(dimensions)) if dimensions else None


def upsert_cached_embeddings(
    path: str | Path,
    embeddings: Iterable[EmbeddingRow],
) -> int:
    """Transactionally insert or replace vectors in an existing cache."""

    prepared: dict[int, np.ndarray] = {}
    for raw_item_id, raw_vector in embeddings:
        item_id = int(raw_item_id)
        if item_id in prepared:
            raise ValueError(f"Duplicate embedding update ID {item_id}")
        prepared[item_id] = _as_vector(item_id, raw_vector)
    if not prepared:
        return 0

    existing_dimension = get_cached_embedding_dimension(path)
    incoming_dimensions = {vector.size for vector in prepared.values()}
    if len(incoming_dimensions) != 1:
        raise ValueError(
            "Embedding updates have inconsistent dimensions: "
            f"{sorted(incoming_dimensions)}"
        )
    incoming_dimension = next(iter(incoming_dimensions))
    if (
        existing_dimension is not None
        and incoming_dimension != existing_dimension
    ):
        raise ValueError(
            f"Embedding update dimension {incoming_dimension} does not match "
            f"cache dimension {existing_dimension}"
        )

    db_path, writable_uri = _existing_database_uri(path, mode="rw")
    rows = [
        (
            item_id,
            json.dumps(vector.tolist(), separators=(",", ":")),
        )
        for item_id, vector in prepared.items()
    ]
    try:
        with sqlite3.connect(writable_uri, uri=True) as connection:
            connection.executemany(
                "INSERT INTO grocery_item_embeddings "
                "(grocery_item_id, embedding) VALUES (?, ?) "
                "ON CONFLICT(grocery_item_id) DO UPDATE SET "
                "embedding = excluded.embedding",
                rows,
            )
    except sqlite3.Error as exc:
        raise RuntimeError(
            f"Could not update embeddings database at {db_path}: {exc}"
        ) from exc

    _clear_memory_cache()
    return len(rows)


def delete_cached_embeddings(
    path: str | Path,
    item_ids: Sequence[int],
) -> int:
    """Transactionally remove item IDs from an existing embedding cache."""

    unique_ids = tuple(dict.fromkeys(int(item_id) for item_id in item_ids))
    if not unique_ids:
        return 0

    # Validate both the file and every existing row before changing the cache.
    read_cached_embeddings(path)
    db_path, writable_uri = _existing_database_uri(path, mode="rw")
    try:
        with sqlite3.connect(writable_uri, uri=True) as connection:
            before_changes = connection.total_changes
            connection.executemany(
                "DELETE FROM grocery_item_embeddings "
                "WHERE grocery_item_id = ?",
                [(item_id,) for item_id in unique_ids],
            )
            deleted_count = connection.total_changes - before_changes
    except sqlite3.Error as exc:
        raise RuntimeError(
            f"Could not update embeddings database at {db_path}: {exc}"
        ) from exc

    _clear_memory_cache()
    return deleted_count


def load_embeddings_into_memory():
    global _cached_path, _cached_vectors

    db_path = get_embeddings_db_path()
    if _cached_vectors is not None and _cached_path == db_path:
        return _cached_vectors

    if not db_path.is_file():
        print(f"[vector_cache] ERROR: Database not found at {db_path}")
        print(
            "[vector_cache] Make sure app.py downloaded it to /tmp "
            "or it exists locally."
        )
        _cached_vectors = []
        _cached_path = db_path
        return _cached_vectors

    print(f"[vector_cache] Loading embeddings from {db_path} ...")

    try:
        _cached_vectors = read_cached_embeddings(db_path)
        print(f"[vector_cache] Loaded {len(_cached_vectors)} vectors into memory")
    except Exception as e:
        print(f"[vector_cache] Database error: {e}")
        _cached_vectors = []
    _cached_path = db_path

    return _cached_vectors


def get_cached_embeddings():
    return load_embeddings_into_memory()
