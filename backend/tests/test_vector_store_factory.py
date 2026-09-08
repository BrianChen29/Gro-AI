import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from vector.factory import (
    close_vector_store,
    get_vector_store,
    set_vector_store_for_testing,
    uses_local_embedding_cache,
    vector_store_backend_name,
)
from vector.in_memory_store import InMemoryVectorStore


class VectorStoreFactoryTests(unittest.TestCase):
    def tearDown(self):
        set_vector_store_for_testing(None)

    def test_memory_is_the_default_and_factory_reuses_one_store(self):
        with patch.dict(os.environ, {}, clear=True):
            first = get_vector_store()
            second = get_vector_store()

        self.assertIsInstance(first, InMemoryVectorStore)
        self.assertIs(first, second)

    def test_backend_name_is_normalized(self):
        with patch.dict(
            os.environ,
            {"VECTOR_STORE_BACKEND": "  In-Memory  "},
        ):
            self.assertEqual(vector_store_backend_name(), "in-memory")
            self.assertIsInstance(get_vector_store(), InMemoryVectorStore)

    def test_pinecone_backend_is_built_from_environment(self):
        pinecone_store = Mock()
        with (
            patch.dict(
                os.environ,
                {"VECTOR_STORE_BACKEND": "pinecone"},
                clear=True,
            ),
            patch(
                "vector.pinecone_store.PineconeVectorStore.from_env",
                return_value=pinecone_store,
            ) as from_env,
        ):
            selected_store = get_vector_store()

        self.assertIs(selected_store, pinecone_store)
        from_env.assert_called_once_with()

    def test_only_memory_backends_use_the_local_embedding_cache(self):
        with patch.dict(
            os.environ,
            {"VECTOR_STORE_BACKEND": "memory"},
            clear=True,
        ):
            self.assertTrue(uses_local_embedding_cache())

        with patch.dict(
            os.environ,
            {"VECTOR_STORE_BACKEND": "pinecone"},
            clear=True,
        ):
            self.assertFalse(uses_local_embedding_cache())

    def test_unknown_backend_has_an_actionable_error(self):
        with patch.dict(
            os.environ,
            {"VECTOR_STORE_BACKEND": "unknown"},
        ):
            with self.assertRaisesRegex(ValueError, "'memory' or 'pinecone'"):
                get_vector_store()


class VectorStoreFactoryLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        set_vector_store_for_testing(None)

    async def test_close_releases_and_forgets_the_selected_store(self):
        selected_store = Mock()
        selected_store.close = AsyncMock()
        set_vector_store_for_testing(selected_store)

        await close_vector_store()

        selected_store.close.assert_awaited_once_with()
        with patch.dict(os.environ, {}, clear=True):
            replacement = get_vector_store()
        self.assertIsInstance(replacement, InMemoryVectorStore)
        self.assertIsNot(replacement, selected_store)


if __name__ == "__main__":
    unittest.main()
