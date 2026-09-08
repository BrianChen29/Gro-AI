from __future__ import annotations

import asyncio
from collections.abc import Sequence

import numpy as np

from vector.factory import get_vector_store
from vector.search_policy import filter_search_hits, get_vector_search_min_score
from vector.store import SearchHit, VectorStore


DEFAULT_MAX_CONCURRENT_SEARCHES = 5
ScoredItem = tuple[int, float]


def _validate_query_embeddings(
    raw_vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
) -> tuple[np.ndarray, ...]:
    if len(raw_vectors) != expected_count:
        raise RuntimeError(
            "Embedding provider returned a different number of query vectors "
            "than requested"
        )

    vectors: list[np.ndarray] = []
    dimensions: set[int] = set()
    for index, raw_vector in enumerate(raw_vectors):
        try:
            vector = np.asarray(raw_vector, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid query embedding at index {index}") from exc
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError(f"Invalid query embedding at index {index}")
        vectors.append(vector)
        dimensions.add(vector.size)

    if len(dimensions) > 1:
        raise ValueError(
            f"Query embeddings have inconsistent dimensions: {sorted(dimensions)}"
        )
    return tuple(vectors)


async def embed_queries(texts: Sequence[str]) -> tuple[np.ndarray, ...]:
    """Embed multiple search queries in one provider request."""

    from vector.embedding_client import get_embeddings

    inputs = list(texts)
    raw_vectors = await get_embeddings(inputs)
    return _validate_query_embeddings(raw_vectors, expected_count=len(inputs))


async def embed_query(text: str) -> np.ndarray:
    """Embed one search query through the shared batch implementation."""

    return (await embed_queries([text]))[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Backward-compatible helper retained for callers and comparisons."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _as_scored_items(
    hits: Sequence[SearchHit],
    min_score: float | None,
) -> list[ScoredItem]:
    filtered_hits = filter_search_hits(hits, min_score)
    return [(hit.item_id, hit.score) for hit in filtered_hits]


async def search_similar_items(
    query: str,
    top_k: int = 10,
) -> list[ScoredItem]:
    """Return top-k matched grocery items by cosine similarity."""

    min_score = get_vector_search_min_score()
    q_emb = await embed_query(query)
    hits = await get_vector_store().search(q_emb, top_k=top_k)
    return _as_scored_items(hits, min_score)


async def _search_with_semaphore(
    store: VectorStore,
    query_vector: np.ndarray,
    *,
    top_k: int,
    min_score: float | None,
    semaphore: asyncio.Semaphore,
) -> list[ScoredItem]:
    async with semaphore:
        hits = await store.search(query_vector, top_k=top_k)
    return _as_scored_items(hits, min_score)


async def search_similar_items_batch(
    queries: Sequence[str],
    top_k: int = 10,
    *,
    max_concurrency: int = DEFAULT_MAX_CONCURRENT_SEARCHES,
) -> list[list[ScoredItem]]:
    """Search multiple queries while bounding concurrent store requests."""

    if (
        isinstance(max_concurrency, bool)
        or not isinstance(max_concurrency, int)
        or max_concurrency <= 0
    ):
        raise ValueError("max_concurrency must be a positive integer")

    inputs = list(queries)
    if not inputs:
        return []

    min_score = get_vector_search_min_score()
    query_vectors = await embed_queries(inputs)
    store = get_vector_store()
    semaphore = asyncio.Semaphore(max_concurrency)
    results = await asyncio.gather(
        *(
            _search_with_semaphore(
                store,
                query_vector,
                top_k=top_k,
                min_score=min_score,
                semaphore=semaphore,
            )
            for query_vector in query_vectors
        )
    )
    return list(results)
