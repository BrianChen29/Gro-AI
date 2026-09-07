"""Plan and execute the initial catalog-vector migration to Pinecone."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from vector.store import VectorRecord, VectorStore
from vector.vector_cache import get_embeddings_db_path, read_cached_embeddings

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class CatalogItemLike(Protocol):
    id: int
    title: str
    sub_category: str


@dataclass(frozen=True, slots=True)
class CatalogSyncPlan:
    """Validated records and reconciliation details for one initial sync."""

    records: tuple[VectorRecord, ...] = field(repr=False)
    catalog_item_count: int
    cached_embedding_count: int
    missing_embedding_ids: tuple[int, ...]
    orphan_embedding_ids: tuple[int, ...]
    vector_dimension: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_item_count": self.catalog_item_count,
            "cached_embedding_count": self.cached_embedding_count,
            "ready_record_count": len(self.records),
            "missing_embedding_count": len(self.missing_embedding_ids),
            "missing_embedding_ids": list(self.missing_embedding_ids),
            "orphan_embedding_count": len(self.orphan_embedding_ids),
            "orphan_embedding_ids": list(self.orphan_embedding_ids),
            "vector_dimension": self.vector_dimension,
        }

    def ensure_ready(self, *, allow_partial: bool = False) -> None:
        if self.catalog_item_count == 0:
            raise RuntimeError("MySQL grocery_items contains no catalog items")
        if not self.records:
            raise RuntimeError("No catalog items have cached embeddings to sync")
        if self.missing_embedding_ids and not allow_partial:
            raise RuntimeError(
                f"{len(self.missing_embedding_ids)} catalog items are missing "
                "cached embeddings; regenerate them or pass --allow-partial"
            )


def _unique_catalog_items(
    items: Iterable[CatalogItemLike],
) -> dict[int, CatalogItemLike]:
    by_id: dict[int, CatalogItemLike] = {}
    for item in items:
        item_id = int(item.id)
        if item_id in by_id:
            raise ValueError(f"Duplicate grocery item ID {item_id}")
        by_id[item_id] = item
    return by_id


def _unique_embeddings(
    embeddings: Iterable[tuple[int, np.ndarray]],
) -> dict[int, np.ndarray]:
    by_id: dict[int, np.ndarray] = {}
    for raw_item_id, raw_vector in embeddings:
        item_id = int(raw_item_id)
        if item_id in by_id:
            raise ValueError(f"Duplicate cached embedding ID {item_id}")

        vector = np.asarray(raw_vector, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError(f"Invalid cached embedding for grocery item {item_id}")
        by_id[item_id] = vector
    return by_id


def build_catalog_sync_plan(
    items: Iterable[CatalogItemLike],
    embeddings: Iterable[tuple[int, np.ndarray]],
) -> CatalogSyncPlan:
    """Join MySQL catalog metadata to cached vectors by grocery item ID."""

    items_by_id = _unique_catalog_items(items)
    embeddings_by_id = _unique_embeddings(embeddings)

    catalog_ids = set(items_by_id)
    embedding_ids = set(embeddings_by_id)
    missing_ids = tuple(sorted(catalog_ids - embedding_ids))
    orphan_ids = tuple(sorted(embedding_ids - catalog_ids))

    records: list[VectorRecord] = []
    dimensions: set[int] = set()
    for item_id in sorted(catalog_ids & embedding_ids):
        item = items_by_id[item_id]
        vector = embeddings_by_id[item_id]
        dimensions.add(vector.size)
        records.append(
            VectorRecord(
                item_id=item_id,
                values=vector,
                metadata={
                    "title": str(item.title),
                    "sub_category": str(item.sub_category),
                },
            )
        )

    if len(dimensions) > 1:
        raise ValueError(
            "Catalog embeddings have inconsistent dimensions: "
            f"{sorted(dimensions)}"
        )

    return CatalogSyncPlan(
        records=tuple(records),
        catalog_item_count=len(items_by_id),
        cached_embedding_count=len(embeddings_by_id),
        missing_embedding_ids=missing_ids,
        orphan_embedding_ids=orphan_ids,
        vector_dimension=(next(iter(dimensions)) if dimensions else None),
    )


async def apply_catalog_sync_plan(
    plan: CatalogSyncPlan,
    store: VectorStore,
    *,
    allow_partial: bool = False,
) -> int:
    """Validate completeness, then upsert all ready records."""

    plan.ensure_ready(allow_partial=allow_partial)
    await store.upsert(plan.records)
    return len(plan.records)


async def load_catalog_items(session: AsyncSession) -> list[CatalogItemLike]:
    """Load source-of-truth catalog rows without coupling pure plan tests to ORM."""

    from sqlalchemy import select

    from db import GroceryItem

    result = await session.execute(select(GroceryItem))
    return list(result.scalars().all())


async def _run_cli(args: argparse.Namespace) -> None:
    embeddings_path = args.embeddings_db or get_embeddings_db_path()
    embeddings = read_cached_embeddings(embeddings_path)

    from db import SessionLocal, engine

    try:
        async with SessionLocal() as session:
            items = await load_catalog_items(session)
    finally:
        await engine.dispose()

    plan = build_catalog_sync_plan(items, embeddings)
    report = {
        "mode": "execute" if args.execute else "dry-run",
        "embeddings_db": str(embeddings_path),
        **plan.as_dict(),
    }

    if not args.execute:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    from vector.factory import (
        close_vector_store,
        get_vector_store,
        vector_store_backend_name,
    )

    if vector_store_backend_name() != "pinecone":
        raise RuntimeError(
            "--execute requires VECTOR_STORE_BACKEND=pinecone"
        )

    store = get_vector_store()
    try:
        report["upserted_count"] = await apply_catalog_sync_plan(
            plan,
            store,
            allow_partial=args.allow_partial,
        )
    finally:
        await close_vector_store()

    print(json.dumps(report, ensure_ascii=False, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or execute the initial Gro AI catalog sync"
    )
    parser.add_argument(
        "--embeddings-db",
        type=Path,
        help="SQLite embedding cache; defaults to EMBEDDINGS_DB_PATH or auto-detection",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write vectors to Pinecone; omission performs a read-only dry run",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow execution even when some MySQL items have no cached vector",
    )
    return parser


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = _build_parser()
    try:
        asyncio.run(_run_cli(parser.parse_args()))
    except Exception as exc:
        parser.exit(1, f"Catalog sync failed: {exc}\n")


if __name__ == "__main__":
    main()
