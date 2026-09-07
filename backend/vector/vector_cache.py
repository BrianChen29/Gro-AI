import json
import os
import sqlite3
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

    db_path = Path(path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Embeddings database not found at {db_path}")

    try:
        read_only_uri = f"{db_path.resolve().as_uri()}?mode=ro"
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
            vector = np.asarray(
                json.loads(raw_embedding),
                dtype=np.float32,
            ).reshape(-1)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid cached embedding for grocery item {item_id}"
            ) from exc

        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError(
                f"Invalid cached embedding for grocery item {item_id}"
            )
        vectors.append((int(item_id), vector))
    return vectors


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
