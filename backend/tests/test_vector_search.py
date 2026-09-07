import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from vector.store import SearchHit
from vector.vector_search import search_similar_items


class VectorSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_public_search_shape_is_preserved_when_using_store(self):
        query_vector = np.array([0.25, 0.75], dtype=np.float32)
        store = Mock()
        store.search = AsyncMock(
            return_value=[
                SearchHit(item_id=11, score=0.93),
                SearchHit(item_id=22, score=0.81),
            ]
        )

        with (
            patch(
                "vector.vector_search.embed_query",
                AsyncMock(return_value=query_vector),
            ) as embed_query,
            patch("vector.vector_search.get_vector_store", return_value=store),
            patch(
                "vector.vector_search.get_vector_search_min_score",
                return_value=None,
            ),
        ):
            result = await search_similar_items("oat milk", top_k=2)

        self.assertEqual(result, [(11, 0.93), (22, 0.81)])
        embed_query.assert_awaited_once_with("oat milk")
        store.search.assert_awaited_once()
        call = store.search.await_args
        np.testing.assert_array_equal(call.args[0], query_vector)
        self.assertEqual(call.kwargs, {"top_k": 2})

    async def test_configured_score_threshold_filters_hits_inclusively(self):
        query_vector = np.array([1.0, 0.0], dtype=np.float32)
        store = Mock()
        store.search = AsyncMock(
            return_value=[
                SearchHit(item_id=11, score=0.91),
                SearchHit(item_id=22, score=0.80),
                SearchHit(item_id=33, score=0.79),
            ]
        )

        with (
            patch(
                "vector.vector_search.embed_query",
                AsyncMock(return_value=query_vector),
            ),
            patch("vector.vector_search.get_vector_store", return_value=store),
            patch(
                "vector.vector_search.get_vector_search_min_score",
                return_value=0.80,
            ),
        ):
            result = await search_similar_items("oat milk", top_k=3)

        self.assertEqual(result, [(11, 0.91), (22, 0.80)])
        store.search.assert_awaited_once()
        self.assertEqual(store.search.await_args.kwargs, {"top_k": 3})


if __name__ == "__main__":
    unittest.main()
