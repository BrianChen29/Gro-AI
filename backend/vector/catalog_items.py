"""Shared catalog-to-vector transformations."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from vector.store import VectorRecord


class CatalogItemLike(Protocol):
    id: int
    title: str
    sub_category: str


def catalog_item_embedding_text(item: CatalogItemLike) -> str:
    """Build the canonical text used for catalog-item embeddings."""

    return f"{item.title} | {item.sub_category}"


def catalog_item_vector_record(
    item: CatalogItemLike,
    values: np.ndarray,
) -> VectorRecord:
    """Attach stable retrieval metadata to one catalog vector."""

    return VectorRecord(
        item_id=int(item.id),
        values=np.asarray(values, dtype=np.float32).reshape(-1),
        metadata={
            "title": str(item.title),
            "sub_category": str(item.sub_category),
        },
    )
