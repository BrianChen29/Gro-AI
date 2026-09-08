import io
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from vector.catalog_sync import (
    _build_parser,
    _run_cli,
    apply_catalog_sync_plan,
    build_catalog_sync_plan,
)


def catalog_item(item_id: int, title: str, sub_category: str):
    return SimpleNamespace(
        id=item_id,
        title=title,
        sub_category=sub_category,
    )


class CatalogSyncPlanTests(unittest.TestCase):
    def test_joins_vectors_to_mysql_metadata_and_reports_differences(self):
        items = [
            catalog_item(2, "Oat Milk", "Beverages"),
            catalog_item(1, "Whole Milk", "Dairy"),
            catalog_item(3, "Missing Vector", "Dairy"),
        ]
        embeddings = [
            (1, np.array([1.0, 0.0])),
            (2, np.array([0.5, 0.5])),
            (99, np.array([0.0, 1.0])),
        ]

        plan = build_catalog_sync_plan(items, embeddings)

        self.assertEqual([record.item_id for record in plan.records], [1, 2])
        self.assertEqual(
            plan.records[0].metadata,
            {"title": "Whole Milk", "sub_category": "Dairy"},
        )
        self.assertEqual(plan.missing_embedding_ids, (3,))
        self.assertEqual(plan.orphan_embedding_ids, (99,))
        self.assertEqual(plan.vector_dimension, 2)

    def test_inconsistent_dimensions_are_rejected_before_remote_write(self):
        items = [
            catalog_item(1, "Whole Milk", "Dairy"),
            catalog_item(2, "Oat Milk", "Beverages"),
        ]
        embeddings = [
            (1, np.array([1.0, 0.0])),
            (2, np.array([0.5, 0.5, 0.0])),
        ]

        with self.assertRaisesRegex(ValueError, "inconsistent dimensions"):
            build_catalog_sync_plan(items, embeddings)

    def test_duplicate_ids_are_rejected(self):
        duplicate_items = [
            catalog_item(1, "Whole Milk", "Dairy"),
            catalog_item(1, "Duplicate", "Dairy"),
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate grocery item ID 1"):
            build_catalog_sync_plan(duplicate_items, [])

        item = catalog_item(1, "Whole Milk", "Dairy")
        duplicate_embeddings = [
            (1, np.array([1.0, 0.0])),
            (1, np.array([0.0, 1.0])),
        ]
        with self.assertRaisesRegex(ValueError, "cached embedding ID 1"):
            build_catalog_sync_plan([item], duplicate_embeddings)

    def test_cli_is_read_only_unless_execute_is_explicit(self):
        args = _build_parser().parse_args([])

        self.assertFalse(args.execute)
        self.assertFalse(args.allow_partial)


class CatalogSyncExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_embeddings_block_write_by_default(self):
        plan = build_catalog_sync_plan(
            [
                catalog_item(1, "Whole Milk", "Dairy"),
                catalog_item(2, "Missing", "Dairy"),
            ],
            [(1, np.array([1.0, 0.0]))],
        )
        store = Mock()
        store.upsert = AsyncMock()

        with self.assertRaisesRegex(RuntimeError, "--allow-partial"):
            await apply_catalog_sync_plan(plan, store)

        store.upsert.assert_not_awaited()

    async def test_allow_partial_upserts_only_ready_records(self):
        plan = build_catalog_sync_plan(
            [
                catalog_item(1, "Whole Milk", "Dairy"),
                catalog_item(2, "Missing", "Dairy"),
            ],
            [(1, np.array([1.0, 0.0]))],
        )
        store = Mock()
        store.upsert = AsyncMock()

        count = await apply_catalog_sync_plan(
            plan,
            store,
            allow_partial=True,
        )

        self.assertEqual(count, 1)
        store.upsert.assert_awaited_once_with(plan.records)

    async def test_cli_defaults_to_a_read_only_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            embeddings_path = Path(directory) / "embeddings.sqlite"
            with sqlite3.connect(embeddings_path) as connection:
                connection.execute(
                    "CREATE TABLE grocery_item_embeddings ("
                    "grocery_item_id INTEGER PRIMARY KEY, "
                    "embedding TEXT NOT NULL)"
                )
                connection.execute(
                    "INSERT INTO grocery_item_embeddings "
                    "(grocery_item_id, embedding) VALUES (?, ?)",
                    (1, json.dumps([1.0, 0.0])),
                )

            class FakeSessionContext:
                async def __aenter__(self):
                    return object()

                async def __aexit__(self, exc_type, exc, traceback):
                    return False

            engine = SimpleNamespace(dispose=AsyncMock())
            fake_db = ModuleType("db")
            fake_db.SessionLocal = FakeSessionContext
            fake_db.engine = engine
            args = SimpleNamespace(
                embeddings_db=embeddings_path,
                execute=False,
                allow_partial=False,
            )
            output = io.StringIO()

            with (
                patch.dict(sys.modules, {"db": fake_db}),
                patch(
                    "vector.catalog_sync.load_catalog_items",
                    AsyncMock(
                        return_value=[catalog_item(1, "Whole Milk", "Dairy")]
                    ),
                ),
                redirect_stdout(output),
            ):
                await _run_cli(args)

        report = json.loads(output.getvalue())
        self.assertEqual(report["mode"], "dry-run")
        self.assertEqual(report["ready_record_count"], 1)
        self.assertNotIn("upserted_count", report)
        engine.dispose.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
