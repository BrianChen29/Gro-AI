import unittest

import numpy as np

from vector.in_memory_store import InMemoryVectorStore
from vector.store import VectorRecord


class InMemoryVectorStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_results_match_the_previous_per_vector_cosine_loop(self):
        random = np.random.default_rng(seed=29)
        vectors = [
            (item_id, random.normal(size=8).astype(np.float32))
            for item_id in range(100, 120)
        ]
        vectors.append((120, np.zeros(8, dtype=np.float32)))
        query = random.normal(size=8).astype(np.float32)

        expected = []
        for item_id, vector in vectors:
            if np.linalg.norm(query) == 0 or np.linalg.norm(vector) == 0:
                score = 0.0
            else:
                score = float(
                    np.dot(query, vector)
                    / (np.linalg.norm(query) * np.linalg.norm(vector))
                )
            expected.append((item_id, score))
        expected.sort(key=lambda item: item[1], reverse=True)

        hits = await InMemoryVectorStore(lambda: vectors).search(query, top_k=10)

        self.assertEqual(
            [hit.item_id for hit in hits],
            [item_id for item_id, _ in expected[:10]],
        )
        np.testing.assert_allclose(
            [hit.score for hit in hits],
            [score for _, score in expected[:10]],
            rtol=1e-5,
            atol=1e-6,
        )

    async def test_search_returns_exact_cosine_order_and_scores(self):
        store = InMemoryVectorStore(
            lambda: [
                (101, np.array([1.0, 0.0], dtype=np.float32)),
                (102, np.array([0.8, 0.2], dtype=np.float32)),
                (103, np.array([0.0, 1.0], dtype=np.float32)),
            ]
        )

        hits = await store.search(np.array([1.0, 0.0]), top_k=2)

        self.assertEqual([hit.item_id for hit in hits], [101, 102])
        self.assertAlmostEqual(hits[0].score, 1.0, places=6)
        self.assertGreater(hits[1].score, 0.9)

    async def test_search_is_stable_when_scores_tie(self):
        store = InMemoryVectorStore(
            lambda: [
                (10, np.array([1.0, 0.0])),
                (20, np.array([1.0, 0.0])),
            ]
        )

        hits = await store.search(np.array([1.0, 0.0]), top_k=2)

        self.assertEqual([hit.item_id for hit in hits], [10, 20])

    async def test_zero_query_matches_previous_zero_score_behavior(self):
        store = InMemoryVectorStore(
            lambda: [
                (10, np.array([1.0, 0.0])),
                (20, np.array([0.0, 1.0])),
            ]
        )

        hits = await store.search(np.array([0.0, 0.0]), top_k=2)

        self.assertEqual([hit.score for hit in hits], [0.0, 0.0])
        self.assertEqual([hit.item_id for hit in hits], [10, 20])

    async def test_empty_store_and_non_positive_top_k_return_no_hits(self):
        store = InMemoryVectorStore()

        self.assertEqual(await store.search(np.array([1.0]), top_k=5), [])
        self.assertEqual(await store.search(np.array([1.0]), top_k=0), [])

    async def test_dimension_mismatch_has_actionable_error(self):
        store = InMemoryVectorStore(lambda: [(10, np.array([1.0, 0.0]))])

        with self.assertRaisesRegex(ValueError, "does not match index dimension"):
            await store.search(np.array([1.0, 0.0, 0.0]), top_k=1)

    async def test_upsert_delete_and_equality_filter(self):
        store = InMemoryVectorStore()
        await store.upsert(
            [
                VectorRecord(
                    item_id=10,
                    values=np.array([1.0, 0.0]),
                    metadata={"sub_category": "Beverages"},
                ),
                VectorRecord(
                    item_id=20,
                    values=np.array([0.9, 0.1]),
                    metadata={"sub_category": "Bakery"},
                ),
            ]
        )

        hits = await store.search(
            np.array([1.0, 0.0]),
            top_k=5,
            filters={"sub_category": "Bakery"},
        )
        self.assertEqual([hit.item_id for hit in hits], [20])

        await store.delete([20])
        self.assertEqual(store.count, 1)
        remaining = await store.search(np.array([1.0, 0.0]), top_k=5)
        self.assertEqual([hit.item_id for hit in remaining], [10])


if __name__ == "__main__":
    unittest.main()
