import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from vector.store import SearchHit
from vector.vector_search import (
    embed_query,
    embed_queries,
    search_similar_items,
    search_similar_items_batch,
)


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


class BatchVectorSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_embeds_all_queries_in_one_request_and_converts_vectors(self):
        get_embeddings = AsyncMock(return_value=[[0.1, 0.9], [0.8, 0.2]])

        with patch("vector.embedding_client.get_embeddings", get_embeddings):
            vectors = await embed_queries(["oat milk", "birthday cake"])

        get_embeddings.assert_awaited_once_with(["oat milk", "birthday cake"])
        self.assertEqual(len(vectors), 2)
        np.testing.assert_array_equal(
            vectors[0],
            np.array([0.1, 0.9], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            vectors[1],
            np.array([0.8, 0.2], dtype=np.float32),
        )

    async def test_single_query_embedding_uses_shared_batch_path(self):
        expected = np.array([0.3, 0.7], dtype=np.float32)
        embed_batch = AsyncMock(return_value=(expected,))

        with patch("vector.vector_search.embed_queries", embed_batch):
            vector = await embed_query("milk")

        embed_batch.assert_awaited_once_with(["milk"])
        np.testing.assert_array_equal(vector, expected)

    async def test_rejects_non_finite_and_inconsistent_query_vectors(self):
        invalid_batches = (
            ([[float("nan"), 0.0]], "Invalid query embedding"),
            ([[1.0, 0.0], [1.0]], "inconsistent dimensions"),
        )

        for raw_vectors, expected_error in invalid_batches:
            with self.subTest(expected_error=expected_error):
                with patch(
                    "vector.embedding_client.get_embeddings",
                    AsyncMock(return_value=raw_vectors),
                ):
                    with self.assertRaisesRegex(ValueError, expected_error):
                        await embed_queries(["first"] * len(raw_vectors))

    async def test_preserves_query_and_hit_order_while_applying_threshold(self):
        query_vectors = tuple(
            np.array([value, 0.0], dtype=np.float32) for value in (1.0, 2.0, 3.0)
        )
        store = Mock()

        async def search(query_vector, *, top_k):
            item_id = int(query_vector[0]) * 10
            await asyncio.sleep((4 - int(query_vector[0])) * 0.001)
            return [
                SearchHit(item_id=item_id, score=0.91),
                SearchHit(item_id=item_id + 1, score=0.80),
                SearchHit(item_id=item_id + 2, score=0.79),
            ]

        store.search = AsyncMock(side_effect=search)
        embed_batch = AsyncMock(return_value=query_vectors)
        queries = ["first", "second", "third"]

        with (
            patch("vector.vector_search.embed_queries", embed_batch),
            patch("vector.vector_search.get_vector_store", return_value=store),
            patch(
                "vector.vector_search.get_vector_search_min_score",
                return_value=0.80,
            ),
        ):
            results = await search_similar_items_batch(
                queries,
                top_k=3,
                max_concurrency=2,
            )

        embed_batch.assert_awaited_once_with(queries)
        self.assertEqual(
            results,
            [
                [(10, 0.91), (11, 0.80)],
                [(20, 0.91), (21, 0.80)],
                [(30, 0.91), (31, 0.80)],
            ],
        )
        self.assertEqual(store.search.await_count, 3)
        self.assertTrue(
            all(call.kwargs == {"top_k": 3} for call in store.search.await_args_list)
        )

    async def test_limits_number_of_concurrent_store_searches(self):
        active_searches = 0
        peak_searches = 0
        two_searches_started = asyncio.Event()
        release_searches = asyncio.Event()
        store = Mock()

        async def blocking_search(query_vector, *, top_k):
            nonlocal active_searches, peak_searches
            active_searches += 1
            peak_searches = max(peak_searches, active_searches)
            if active_searches == 2:
                two_searches_started.set()
            try:
                await release_searches.wait()
                return [SearchHit(item_id=int(query_vector[0]), score=0.90)]
            finally:
                active_searches -= 1

        store.search = AsyncMock(side_effect=blocking_search)
        query_vectors = tuple(
            np.array([item_id], dtype=np.float32) for item_id in range(1, 5)
        )

        with (
            patch(
                "vector.vector_search.embed_queries",
                AsyncMock(return_value=query_vectors),
            ),
            patch("vector.vector_search.get_vector_store", return_value=store),
            patch(
                "vector.vector_search.get_vector_search_min_score",
                return_value=None,
            ),
        ):
            batch_task = asyncio.create_task(
                search_similar_items_batch(
                    ["one", "two", "three", "four"],
                    max_concurrency=2,
                )
            )
            await asyncio.wait_for(two_searches_started.wait(), timeout=1)
            self.assertEqual(store.search.await_count, 2)
            release_searches.set()
            results = await batch_task

        self.assertEqual(peak_searches, 2)
        self.assertEqual(store.search.await_count, 4)
        self.assertEqual(results, [[(1, 0.90)], [(2, 0.90)], [(3, 0.90)], [(4, 0.90)]])

    async def test_invalid_embedding_batch_stops_before_store_creation(self):
        store_factory = Mock()

        with (
            patch(
                "vector.embedding_client.get_embeddings",
                AsyncMock(return_value=[[1.0, 0.0]]),
            ),
            patch("vector.vector_search.get_vector_store", store_factory),
            patch(
                "vector.vector_search.get_vector_search_min_score",
                return_value=None,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "different number"):
                await search_similar_items_batch(["first", "second"])

        store_factory.assert_not_called()

    async def test_empty_batch_does_not_initialize_dependencies(self):
        embed_batch = AsyncMock()
        store_factory = Mock()

        with (
            patch("vector.vector_search.embed_queries", embed_batch),
            patch("vector.vector_search.get_vector_store", store_factory),
        ):
            results = await search_similar_items_batch([])

        self.assertEqual(results, [])
        embed_batch.assert_not_awaited()
        store_factory.assert_not_called()

    async def test_rejects_invalid_concurrency_before_starting_work(self):
        for max_concurrency in (0, -1, True, 1.5):
            with self.subTest(max_concurrency=max_concurrency):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    await search_similar_items_batch(
                        ["milk"],
                        max_concurrency=max_concurrency,
                    )


if __name__ == "__main__":
    unittest.main()
