import os
import unittest
from unittest.mock import patch

from vector.factory import (
    get_vector_store,
    set_vector_store_for_testing,
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

    def test_unknown_backend_has_an_actionable_error(self):
        with patch.dict(
            os.environ,
            {"VECTOR_STORE_BACKEND": "unknown"},
        ):
            with self.assertRaisesRegex(ValueError, "supports 'memory'"):
                get_vector_store()


if __name__ == "__main__":
    unittest.main()
