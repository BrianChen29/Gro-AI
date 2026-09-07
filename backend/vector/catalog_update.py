"""Dry-run and execute targeted catalog-vector updates."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Awaitable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from vector.catalog_items import (
    CatalogItemLike,
    catalog_item_embedding_text,
    catalog_item_vector_record,
)
from vector.store import VectorRecord, VectorStore
from vector.vector_cache import (
    delete_cached_embeddings,
    get_embeddings_db_path,
    read_cached_embeddings,
    upsert_cached_embeddings,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


EmbeddingBatchFunction = Callable[
    [Sequence[str]],
    Awaitable[Sequence[Sequence[float]]],
]


def _normalize_item_ids(item_ids: Iterable[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_item_id in item_ids:
        item_id = int(raw_item_id)
        if item_id <= 0:
            raise ValueError("Catalog item IDs must be positive integers")
        if item_id not in seen:
            normalized.append(item_id)
            seen.add(item_id)
    if not normalized:
        raise ValueError("At least one catalog item ID is required")
    return tuple(normalized)


def _items_by_id(
    items: Iterable[CatalogItemLike],
) -> dict[int, CatalogItemLike]:
    by_id: dict[int, CatalogItemLike] = {}
    for item in items:
        item_id = int(item.id)
        if item_id in by_id:
            raise ValueError(f"Duplicate grocery item ID {item_id}")
        by_id[item_id] = item
    return by_id


@dataclass(frozen=True, slots=True)
class CatalogUpsertPlan:
    """Catalog rows selected for targeted re-embedding."""

    requested_item_ids: tuple[int, ...]
    items: tuple[CatalogItemLike, ...] = field(repr=False)
    missing_item_ids: tuple[int, ...]

    @property
    def found_item_ids(self) -> tuple[int, ...]:
        return tuple(int(item.id) for item in self.items)

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_item_ids": list(self.requested_item_ids),
            "target_item_count": len(self.items),
            "target_item_ids": list(self.found_item_ids),
            "missing_item_ids": list(self.missing_item_ids),
            "ready_to_execute": not self.missing_item_ids,
        }

    def ensure_ready(self) -> None:
        if self.missing_item_ids:
            raise RuntimeError(
                "Cannot upsert missing MySQL grocery item IDs: "
                f"{list(self.missing_item_ids)}"
            )


@dataclass(frozen=True, slots=True)
class CatalogDeletePlan:
    """Vector IDs selected for deletion after their source rows are gone."""

    requested_item_ids: tuple[int, ...]
    existing_item_ids: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_item_ids": list(self.requested_item_ids),
            "delete_item_count": len(self.requested_item_ids),
            "blocked_by_existing_item_ids": list(self.existing_item_ids),
            "ready_to_execute": not self.existing_item_ids,
        }

    def ensure_ready(self) -> None:
        if self.existing_item_ids:
            raise RuntimeError(
                "Refusing to delete vectors for grocery items that still exist "
                f"in MySQL: {list(self.existing_item_ids)}"
            )


@dataclass(frozen=True, slots=True)
class CatalogUpsertResult:
    upserted_count: int
    cached_count: int
    vector_dimension: int


@dataclass(frozen=True, slots=True)
class CatalogDeleteResult:
    deleted_count: int
    cache_deleted_count: int


def build_catalog_upsert_plan(
    requested_item_ids: Iterable[int],
    items: Iterable[CatalogItemLike],
) -> CatalogUpsertPlan:
    requested_ids = _normalize_item_ids(requested_item_ids)
    items_by_id = _items_by_id(items)
    unexpected_ids = tuple(
        sorted(set(items_by_id).difference(requested_ids))
    )
    if unexpected_ids:
        raise ValueError(
            f"Loaded unrequested grocery item IDs: {list(unexpected_ids)}"
        )

    ordered_items = tuple(
        items_by_id[item_id]
        for item_id in requested_ids
        if item_id in items_by_id
    )
    missing_ids = tuple(
        item_id for item_id in requested_ids if item_id not in items_by_id
    )
    return CatalogUpsertPlan(
        requested_item_ids=requested_ids,
        items=ordered_items,
        missing_item_ids=missing_ids,
    )


def build_catalog_delete_plan(
    requested_item_ids: Iterable[int],
    existing_items: Iterable[CatalogItemLike],
) -> CatalogDeletePlan:
    requested_ids = _normalize_item_ids(requested_item_ids)
    items_by_id = _items_by_id(existing_items)
    unexpected_ids = tuple(
        sorted(set(items_by_id).difference(requested_ids))
    )
    if unexpected_ids:
        raise ValueError(
            f"Loaded unrequested grocery item IDs: {list(unexpected_ids)}"
        )
    return CatalogDeletePlan(
        requested_item_ids=requested_ids,
        existing_item_ids=tuple(
            item_id for item_id in requested_ids if item_id in items_by_id
        ),
    )


def _cache_dimension(
    embeddings: Sequence[tuple[int, np.ndarray]],
) -> int | None:
    dimensions = {vector.size for _, vector in embeddings}
    if len(dimensions) > 1:
        raise ValueError(
            "Cached embeddings have inconsistent dimensions: "
            f"{sorted(dimensions)}"
        )
    return next(iter(dimensions)) if dimensions else None


async def _embed_catalog_items(
    plan: CatalogUpsertPlan,
    embed_batch: EmbeddingBatchFunction,
) -> tuple[VectorRecord, ...]:
    plan.ensure_ready()
    texts = [catalog_item_embedding_text(item) for item in plan.items]
    raw_vectors = list(await embed_batch(texts))
    if len(raw_vectors) != len(plan.items):
        raise RuntimeError(
            "Embedding provider returned a different number of vectors than "
            "requested"
        )

    records: list[VectorRecord] = []
    dimensions: set[int] = set()
    for item, raw_vector in zip(plan.items, raw_vectors, strict=True):
        vector = np.asarray(raw_vector, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError(
                f"Invalid generated embedding for grocery item {item.id}"
            )
        dimensions.add(vector.size)
        records.append(catalog_item_vector_record(item, vector))

    if len(dimensions) != 1:
        raise ValueError(
            "Generated embeddings have inconsistent dimensions: "
            f"{sorted(dimensions)}"
        )
    return tuple(records)


async def apply_catalog_upserts(
    plan: CatalogUpsertPlan,
    store: VectorStore,
    embeddings_path: str | Path,
    embed_batch: EmbeddingBatchFunction,
) -> CatalogUpsertResult:
    """Re-embed selected rows, then update the remote store and local cache."""

    plan.ensure_ready()
    cached_embeddings = read_cached_embeddings(embeddings_path)
    cache_dimension = _cache_dimension(cached_embeddings)
    records = await _embed_catalog_items(plan, embed_batch)
    vector_dimension = records[0].values.size
    if cache_dimension is not None and vector_dimension != cache_dimension:
        raise ValueError(
            f"Generated embedding dimension {vector_dimension} does not match "
            f"cache dimension {cache_dimension}"
        )

    await store.upsert(records)
    try:
        cached_count = upsert_cached_embeddings(
            embeddings_path,
            [(record.item_id, record.values) for record in records],
        )
    except Exception as exc:
        raise RuntimeError(
            "Vector-store upsert succeeded, but the SQLite cache update failed; "
            "rerun the same command to reconcile them"
        ) from exc
    return CatalogUpsertResult(
        upserted_count=len(records),
        cached_count=cached_count,
        vector_dimension=vector_dimension,
    )


async def apply_catalog_deletes(
    plan: CatalogDeletePlan,
    store: VectorStore,
    embeddings_path: str | Path,
) -> CatalogDeleteResult:
    """Delete selected IDs from the remote store and local cache."""

    plan.ensure_ready()
    read_cached_embeddings(embeddings_path)
    await store.delete(plan.requested_item_ids)
    try:
        cache_deleted_count = delete_cached_embeddings(
            embeddings_path,
            plan.requested_item_ids,
        )
    except Exception as exc:
        raise RuntimeError(
            "Vector-store deletion succeeded, but the SQLite cache update "
            "failed; rerun the same command to reconcile them"
        ) from exc
    return CatalogDeleteResult(
        deleted_count=len(plan.requested_item_ids),
        cache_deleted_count=cache_deleted_count,
    )


async def load_catalog_items_by_ids(
    session: AsyncSession,
    item_ids: Sequence[int],
) -> list[CatalogItemLike]:
    """Load only requested source-of-truth rows from MySQL."""

    from sqlalchemy import select

    from db import GroceryItem

    result = await session.execute(
        select(GroceryItem).where(GroceryItem.id.in_(item_ids))
    )
    return list(result.scalars().all())


async def _run_cli(args: argparse.Namespace) -> None:
    item_ids = _normalize_item_ids(args.item_ids)
    embeddings_path = args.embeddings_db or get_embeddings_db_path()
    cached_embeddings = read_cached_embeddings(embeddings_path)
    cache_dimension = _cache_dimension(cached_embeddings)

    from db import SessionLocal, engine

    try:
        async with SessionLocal() as session:
            items = await load_catalog_items_by_ids(session, item_ids)
    finally:
        await engine.dispose()

    report: dict[str, object] = {
        "mode": "execute" if args.execute else "dry-run",
        "operation": args.operation,
        "embeddings_db": str(embeddings_path),
        "cache_vector_count": len(cached_embeddings),
        "cache_vector_dimension": cache_dimension,
    }

    if args.operation == "upsert":
        plan = build_catalog_upsert_plan(item_ids, items)
        from vector.embedding_client import get_embedding_model

        report.update(
            {
                "embedding_model": get_embedding_model(),
                **plan.as_dict(),
            }
        )
    else:
        plan = build_catalog_delete_plan(item_ids, items)
        report.update(plan.as_dict())

    if not args.execute:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    plan.ensure_ready()

    from vector.factory import (
        close_vector_store,
        get_vector_store,
        vector_store_backend_name,
    )

    if vector_store_backend_name() != "pinecone":
        raise RuntimeError("--execute requires VECTOR_STORE_BACKEND=pinecone")

    store = get_vector_store()
    try:
        if args.operation == "upsert":
            from vector.embedding_client import get_embeddings

            upsert_result = await apply_catalog_upserts(
                plan,
                store,
                embeddings_path,
                get_embeddings,
            )
            report.update(
                {
                    "upserted_count": upsert_result.upserted_count,
                    "cache_upserted_count": upsert_result.cached_count,
                    "vector_dimension": upsert_result.vector_dimension,
                }
            )
        else:
            delete_result = await apply_catalog_deletes(
                plan,
                store,
                embeddings_path,
            )
            report.update(
                {
                    "deleted_count": delete_result.deleted_count,
                    "cache_deleted_count": delete_result.cache_deleted_count,
                }
            )
    finally:
        await close_vector_store()

    print(json.dumps(report, ensure_ascii=False, indent=2))


def _positive_item_id(raw_value: str) -> int:
    try:
        item_id = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("item IDs must be integers") from exc
    if item_id <= 0:
        raise argparse.ArgumentTypeError("item IDs must be positive")
    return item_id


def _add_operation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "item_ids",
        nargs="+",
        type=_positive_item_id,
        metavar="ITEM_ID",
        help="one or more grocery item IDs",
    )
    parser.add_argument(
        "--embeddings-db",
        type=Path,
        help="SQLite embedding cache; defaults to EMBEDDINGS_DB_PATH",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="apply changes; omission performs a read-only dry run",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or execute targeted Gro AI catalog-vector updates"
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    upsert_parser = subparsers.add_parser(
        "upsert",
        help="re-embed existing MySQL catalog items",
    )
    _add_operation_arguments(upsert_parser)

    delete_parser = subparsers.add_parser(
        "delete",
        help="remove vectors after their MySQL rows are deleted",
    )
    _add_operation_arguments(delete_parser)
    return parser


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = _build_parser()
    try:
        asyncio.run(_run_cli(parser.parse_args()))
    except Exception as exc:
        parser.exit(1, f"Catalog update failed: {exc}\n")


if __name__ == "__main__":
    main()
