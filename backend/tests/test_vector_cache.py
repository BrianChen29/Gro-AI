import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from vector.vector_cache import (
    get_embeddings_db_path,
    read_cached_embeddings,
)


class VectorCacheTests(unittest.TestCase):
    def _create_cache(self, path: Path, rows: list[tuple[int, str]]) -> None:
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE grocery_item_embeddings ("
                "grocery_item_id INTEGER PRIMARY KEY, "
                "embedding TEXT NOT NULL)"
            )
            connection.executemany(
                "INSERT INTO grocery_item_embeddings "
                "(grocery_item_id, embedding) VALUES (?, ?)",
                rows,
            )

    def test_reads_valid_embeddings_from_an_explicit_database(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.sqlite"
            self._create_cache(
                path,
                [
                    (7, json.dumps([0.1, 0.2])),
                    (8, json.dumps([0.3, 0.4])),
                ],
            )

            rows = read_cached_embeddings(path)

        self.assertEqual([item_id for item_id, _ in rows], [7, 8])
        np.testing.assert_allclose(rows[0][1], [0.1, 0.2])

    def test_missing_database_and_invalid_rows_fail_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            missing_path = directory_path / "missing.sqlite"
            with self.assertRaisesRegex(FileNotFoundError, "not found"):
                read_cached_embeddings(missing_path)

            invalid_path = directory_path / "invalid.sqlite"
            self._create_cache(invalid_path, [(7, "not-json")])
            with self.assertRaisesRegex(ValueError, "grocery item 7"):
                read_cached_embeddings(invalid_path)

    def test_configured_path_takes_precedence(self):
        configured = "/tmp/configured-groai-embeddings.sqlite"
        with patch.dict(
            os.environ,
            {"EMBEDDINGS_DB_PATH": configured},
            clear=True,
        ):
            self.assertEqual(get_embeddings_db_path(), Path(configured))


if __name__ == "__main__":
    unittest.main()
