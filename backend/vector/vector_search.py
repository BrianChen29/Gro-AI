import numpy as np

from vector.factory import get_vector_store
from vector.search_policy import filter_search_hits, get_vector_search_min_score


async def embed_query(text: str):
    """LLM embedding for the search query"""
    # Keep the provider import at the integration boundary so the vector-store
    # layer can be unit-tested without initializing external LLM SDKs.
    from vector.embedding_client import get_embedding

    return np.array(await get_embedding(text), dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Backward-compatible helper retained for callers and comparisons."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


async def search_similar_items(query: str, top_k: int = 10):
    """Return top-k matched grocery items by cosine similarity."""
    q_emb = await embed_query(query)
    hits = await get_vector_store().search(q_emb, top_k=top_k)
    filtered_hits = filter_search_hits(hits, get_vector_search_min_score())
    return [(hit.item_id, hit.score) for hit in filtered_hits]
