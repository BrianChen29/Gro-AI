import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from vector.embedding_client import get_embedding, get_embeddings


class EmbeddingClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_results_are_restored_to_input_order(self):
        create = AsyncMock(
            return_value=SimpleNamespace(
                data=[
                    SimpleNamespace(index=1, embedding=[0.2, 0.8]),
                    SimpleNamespace(index=0, embedding=[0.9, 0.1]),
                ]
            )
        )
        client = SimpleNamespace(embeddings=SimpleNamespace(create=create))

        with (
            patch("vector.embedding_client._new_client", return_value=client),
            patch.dict(
                os.environ,
                {"OPENAI_EMBEDDING_MODEL": "configured-embedding-model"},
            ),
        ):
            vectors = await get_embeddings(["first", "second"])

        self.assertEqual(vectors, [[0.9, 0.1], [0.2, 0.8]])
        create.assert_awaited_once_with(
            model="configured-embedding-model",
            input=["first", "second"],
        )

    async def test_single_embedding_uses_batch_helper(self):
        with patch(
            "vector.embedding_client.get_embeddings",
            AsyncMock(return_value=[[0.3, 0.7]]),
        ) as get_batch:
            vector = await get_embedding("milk", model="test-model")

        self.assertEqual(vector, [0.3, 0.7])
        get_batch.assert_awaited_once_with(["milk"], model="test-model")

    async def test_empty_batch_returns_without_creating_client(self):
        client_factory = Mock()
        with patch("vector.embedding_client._new_client", client_factory):
            vectors = await get_embeddings([])

        self.assertEqual(vectors, [])
        client_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
