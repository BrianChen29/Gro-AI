"""Shared contracts for catalog vector-store backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

import numpy as np


MetadataValue = str | int | float | bool | list[str]
VectorFilter = Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A catalog item returned by a vector similarity search."""

    item_id: int
    score: float
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """A catalog vector and the metadata needed for filtering or debugging."""

    item_id: int
    values: np.ndarray = field(repr=False)
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


class VectorStore(Protocol):
    """Backend-neutral operations used by Gro AI's catalog grounding layer."""

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: VectorFilter | None = None,
    ) -> list[SearchHit]:
        """Return the most similar catalog items, ordered best first."""
        ...

    async def upsert(self, records: Sequence[VectorRecord]) -> None:
        """Insert new vectors or replace existing vectors with the same IDs."""
        ...

    async def delete(self, item_ids: Sequence[int]) -> None:
        """Delete catalog vectors by item ID."""
        ...

    async def close(self) -> None:
        """Release backend resources. In-memory implementations may do nothing."""
        ...
