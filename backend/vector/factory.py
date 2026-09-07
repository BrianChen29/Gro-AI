"""Configuration and lifecycle helpers for the selected vector-store backend."""

from __future__ import annotations

import os

from vector.in_memory_store import InMemoryVectorStore
from vector.store import VectorStore
from vector.vector_cache import get_cached_embeddings


MEMORY_BACKEND_NAMES = frozenset({"memory", "in-memory", "exact"})
_vector_store: VectorStore | None = None


def vector_store_backend_name() -> str:
    return os.getenv("VECTOR_STORE_BACKEND", "memory").strip().lower()


def uses_local_embedding_cache() -> bool:
    return vector_store_backend_name() in MEMORY_BACKEND_NAMES


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    backend = vector_store_backend_name()
    if backend in MEMORY_BACKEND_NAMES:
        _vector_store = InMemoryVectorStore(get_cached_embeddings)
    elif backend == "pinecone":
        from vector.pinecone_store import PineconeVectorStore

        _vector_store = PineconeVectorStore.from_env()
    else:
        raise ValueError(
            f"Unsupported VECTOR_STORE_BACKEND={backend!r}; "
            "expected 'memory' or 'pinecone'"
        )
    return _vector_store


async def close_vector_store() -> None:
    global _vector_store
    if _vector_store is None:
        return
    await _vector_store.close()
    _vector_store = None


def set_vector_store_for_testing(store: VectorStore | None) -> None:
    """Replace the process-wide store without touching external resources."""

    global _vector_store
    _vector_store = store
