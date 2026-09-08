import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from vector.pinecone_store import PineconeVectorStore
from vector.store import VectorRecord


class FakePineconeIndex:
    def __init__(self):
        self.query_request = None
        self.upsert_request = None
        self.delete_request = None
        self.upsert_response = None
        self.closed = False

    async def query(self, **kwargs):
        self.query_request = kwargs
        return {
            "matches": [
                {
                    "id": "42",
                    "score": 0.91,
                    "metadata": {"sub_category": "Beverages"},
                }
            ]
        }

    async def upsert(self, **kwargs):
        self.upsert_request = kwargs
        return self.upsert_response

    async def delete(self, **kwargs):
        self.delete_request = kwargs

    async def close(self):
        self.closed = True


class PineconeVectorStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_maps_query_filter_and_response(self):
        index = FakePineconeIndex()
        store = PineconeVectorStore(index=index, namespace="supplier-7")

        hits = await store.search(
            np.array([0.1, 0.2], dtype=np.float32),
            top_k=3,
            filters={"active": True},
        )

        self.assertEqual(hits[0].item_id, 42)
        self.assertAlmostEqual(hits[0].score, 0.91)
        self.assertEqual(hits[0].metadata["sub_category"], "Beverages")
        self.assertEqual(index.query_request["namespace"], "supplier-7")
        self.assertEqual(index.query_request["top_k"], 3)
        self.assertEqual(index.query_request["filter"], {"active": True})
        self.assertFalse(index.query_request["include_values"])
        self.assertTrue(index.query_request["include_metadata"])

    async def test_upsert_maps_catalog_ids_metadata_and_batch_settings(self):
        index = FakePineconeIndex()
        store = PineconeVectorStore(
            index=index,
            namespace="catalog",
            batch_size=25,
        )

        await store.upsert(
            [
                VectorRecord(
                    item_id=7,
                    values=np.array([1.0, 2.0], dtype=np.float32),
                    metadata={"title": "Whole Milk"},
                )
            ]
        )

        self.assertEqual(index.upsert_request["namespace"], "catalog")
        self.assertEqual(index.upsert_request["batch_size"], 25)
        self.assertFalse(index.upsert_request["show_progress"])
        self.assertEqual(
            index.upsert_request["vectors"],
            [
                {
                    "id": "7",
                    "values": [1.0, 2.0],
                    "metadata": {"title": "Whole Milk"},
                }
            ],
        )

    async def test_upsert_surfaces_partial_batch_failures(self):
        index = FakePineconeIndex()
        index.upsert_response = SimpleNamespace(
            has_errors=True,
            errors=["batch 2 timed out"],
        )
        store = PineconeVectorStore(index=index)

        with self.assertRaisesRegex(RuntimeError, "partial failures"):
            await store.upsert(
                [VectorRecord(item_id=7, values=np.array([1.0, 2.0]))]
            )

    async def test_delete_and_close(self):
        index = FakePineconeIndex()
        store = PineconeVectorStore(index=index, namespace="catalog")

        await store.delete([7, 8])
        await store.close()

        self.assertEqual(
            index.delete_request,
            {"ids": ["7", "8"], "namespace": "catalog"},
        )
        self.assertTrue(index.closed)

    async def test_invalid_or_non_catalog_vectors_are_rejected(self):
        index = FakePineconeIndex()
        store = PineconeVectorStore(index=index)

        with self.assertRaisesRegex(ValueError, "finite numbers"):
            await store.search(np.array([1.0, np.nan]), top_k=1)

        async def invalid_query(**kwargs):
            return {"matches": [{"id": "not-a-number", "score": 1.0}]}

        index.query = invalid_query
        with self.assertRaisesRegex(ValueError, "non-catalog vector ID"):
            await store.search(np.array([1.0, 0.0]), top_k=1)

    def test_environment_configuration_is_validated(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "PINECONE_API_KEY"):
                PineconeVectorStore.from_env()

        with patch.dict(
            os.environ,
            {
                "PINECONE_API_KEY": "test-key",
                "PINECONE_INDEX_HOST": "test-host",
                "PINECONE_BATCH_SIZE": "not-an-integer",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                PineconeVectorStore.from_env()


if __name__ == "__main__":
    unittest.main()
