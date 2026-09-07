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

from vector.catalog_update import (
    _build_parser,
    _run_cli,
    apply_catalog_deletes,
    apply_catalog_upserts,
    build_catalog_delete_plan,
    build_catalog_upsert_plan,
)
from vector.vector_cache import read_cached_embeddings


def catalog_item(item_id: int, title: str, sub_category: str):
    return SimpleNamespace(
        id=item_id,
        title=title,
        sub_category=sub_category,
    )


def create_cache(path: Path, rows: list[tuple[int, list[float]]]) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE grocery_item_embeddings ("
            "grocery_item_id INTEGER PRIMARY KEY, "
            "embedding TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO grocery_item_embeddings "
            "(grocery_item_id, embedding) VALUES (?, ?)",
            [(item_id, json.dumps(vector)) for item_id, vector in rows],
        )


class CatalogUpdatePlanTests(unittest.TestCase):
    def test_upsert_plan_preserves_requested_order_and_reports_missing_ids(self):
        plan = build_catalog_upsert_plan(
            [2, 1, 3, 2],
            [
                catalog_item(1, "Whole Milk", "Dairy"),
                catalog_item(2, "Oat Milk", "Beverages"),
            ],
        )

        self.assertEqual(plan.requested_item_ids, (2, 1, 3))
        self.assertEqual(plan.found_item_ids, (2, 1))
        self.assertEqual(plan.missing_item_ids, (3,))
        with self.assertRaisesRegex(RuntimeError, "missing MySQL"):
            plan.ensure_ready()

    def test_delete_plan_blocks_ids_that_still_exist_in_mysql(self):
        plan = build_catalog_delete_plan(
            [1, 2],
            [catalog_item(2, "Oat Milk", "Beverages")],
        )

        self.assertEqual(plan.existing_item_ids, (2,))
        with self.assertRaisesRegex(RuntimeError, "still exist"):
            plan.ensure_ready()

    def test_cli_defaults_to_dry_run_for_both_operations(self):
        parser = _build_parser()

        upsert_args = parser.parse_args(["upsert", "1", "2"])
        delete_args = parser.parse_args(["delete", "3"])

        self.assertFalse(upsert_args.execute)
        self.assertFalse(delete_args.execute)


class CatalogUpdateExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_upsert_embeds_in_order_and_updates_store_and_cache(self):
        plan = build_catalog_upsert_plan(
            [2, 1],
            [
                catalog_item(1, "Whole Milk", "Dairy"),
                catalog_item(2, "Oat Milk", "Beverages"),
            ],
        )
        embed_batch = AsyncMock(
            return_value=[[0.2, 0.8], [0.9, 0.1]]
        )
        store = Mock()
        store.upsert = AsyncMock()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.sqlite"
            create_cache(path, [(1, [0.1, 0.9])])

            result = await apply_catalog_upserts(
                plan,
                store,
                path,
                embed_batch,
            )
            cached_rows = dict(read_cached_embeddings(path))

        embed_batch.assert_awaited_once_with(
            ["Oat Milk | Beverages", "Whole Milk | Dairy"]
        )
        records = store.upsert.await_args.args[0]
        self.assertEqual([record.item_id for record in records], [2, 1])
        self.assertEqual(records[0].metadata["title"], "Oat Milk")
        np.testing.assert_allclose(cached_rows[2], [0.2, 0.8])
        np.testing.assert_allclose(cached_rows[1], [0.9, 0.1])
        self.assertEqual(result.upserted_count, 2)
        self.assertEqual(result.cached_count, 2)
        self.assertEqual(result.vector_dimension, 2)

    async def test_dimension_mismatch_stops_before_remote_write(self):
        plan = build_catalog_upsert_plan(
            [1],
            [catalog_item(1, "Whole Milk", "Dairy")],
        )
        embed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        store = Mock()
        store.upsert = AsyncMock()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.sqlite"
            create_cache(path, [(1, [0.5, 0.5])])

            with self.assertRaisesRegex(ValueError, "does not match"):
                await apply_catalog_upserts(
                    plan,
                    store,
                    path,
                    embed_batch,
                )
            cached_rows = dict(read_cached_embeddings(path))

        store.upsert.assert_not_awaited()
        np.testing.assert_allclose(cached_rows[1], [0.5, 0.5])

    async def test_remote_upsert_failure_leaves_cache_unchanged(self):
        plan = build_catalog_upsert_plan(
            [1],
            [catalog_item(1, "Whole Milk", "Dairy")],
        )
        store = Mock()
        store.upsert = AsyncMock(side_effect=RuntimeError("remote failed"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.sqlite"
            create_cache(path, [(1, [0.5, 0.5])])

            with self.assertRaisesRegex(RuntimeError, "remote failed"):
                await apply_catalog_upserts(
                    plan,
                    store,
                    path,
                    AsyncMock(return_value=[[0.9, 0.1]]),
                )
            cached_rows = dict(read_cached_embeddings(path))

        np.testing.assert_allclose(cached_rows[1], [0.5, 0.5])

    async def test_delete_removes_remote_and_cached_vector(self):
        plan = build_catalog_delete_plan([2], [])
        store = Mock()
        store.delete = AsyncMock()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.sqlite"
            create_cache(path, [(1, [0.5, 0.5]), (2, [0.2, 0.8])])

            result = await apply_catalog_deletes(plan, store, path)
            cached_rows = dict(read_cached_embeddings(path))

        store.delete.assert_awaited_once_with((2,))
        self.assertNotIn(2, cached_rows)
        self.assertIn(1, cached_rows)
        self.assertEqual(result.deleted_count, 1)
        self.assertEqual(result.cache_deleted_count, 1)

    async def test_existing_mysql_item_blocks_delete_before_remote_write(self):
        plan = build_catalog_delete_plan(
            [2],
            [catalog_item(2, "Oat Milk", "Beverages")],
        )
        store = Mock()
        store.delete = AsyncMock()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.sqlite"
            create_cache(path, [(2, [0.2, 0.8])])

            with self.assertRaisesRegex(RuntimeError, "still exist"):
                await apply_catalog_deletes(plan, store, path)

        store.delete.assert_not_awaited()

    async def test_upsert_cli_dry_run_does_not_call_embedding_or_store(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.sqlite"
            create_cache(path, [(1, [0.5, 0.5])])

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
                operation="upsert",
                item_ids=[1],
                embeddings_db=path,
                execute=False,
            )
            output = io.StringIO()

            with (
                patch.dict(sys.modules, {"db": fake_db}),
                patch(
                    "vector.catalog_update.load_catalog_items_by_ids",
                    AsyncMock(
                        return_value=[catalog_item(1, "Whole Milk", "Dairy")]
                    ),
                ),
                patch(
                    "vector.embedding_client.get_embedding_model",
                    return_value="test-model",
                ),
                redirect_stdout(output),
            ):
                await _run_cli(args)

        report = json.loads(output.getvalue())
        self.assertEqual(report["mode"], "dry-run")
        self.assertEqual(report["target_item_ids"], [1])
        self.assertEqual(report["embedding_model"], "test-model")
        engine.dispose.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
