"""Exact cosine vector search for the current small Gro AI catalog."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np

from vector.store import SearchHit, VectorFilter, VectorRecord


RawVector = tuple[int, np.ndarray]
VectorProvider = Callable[[], Iterable[RawVector]]


class InMemoryVectorStore:
    """Lazy, exact cosine search over a normalized in-memory matrix.

    This preserves the existing exhaustive-search behavior while avoiding the
    previous Python loop and repeated norm calculation for every catalog item.
    """

    def __init__(self, vectors_provider: VectorProvider | None = None) -> None:
        self._vectors_provider = vectors_provider or (lambda: ())
        self._records: dict[int, VectorRecord] | None = None
        self._item_ids = np.empty(0, dtype=np.int64)
        self._normalized_vectors = np.empty((0, 0), dtype=np.float32)

    @property
    def count(self) -> int:
        self._ensure_loaded()
        return len(self._item_ids)

    def _ensure_loaded(self) -> None:
        if self._records is not None:
            return

        records: dict[int, VectorRecord] = {}
        for item_id, values in self._vectors_provider():
            records[int(item_id)] = VectorRecord(
                item_id=int(item_id),
                values=self._as_vector(values),
            )

        self._records = records
        try:
            self._rebuild_index()
        except Exception:
            # Do not leave a half-loaded index behind. A later call may retry
            # after the source data has been corrected.
            self._records = None
            self._item_ids = np.empty(0, dtype=np.int64)
            self._normalized_vectors = np.empty((0, 0), dtype=np.float32)
            raise

    @staticmethod
    def _as_vector(values: np.ndarray) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise ValueError("Vector values must not be empty")
        if not np.all(np.isfinite(vector)):
            raise ValueError("Vector values must contain only finite numbers")
        return vector.copy()

    def _rebuild_index(self) -> None:
        assert self._records is not None

        if not self._records:
            self._item_ids = np.empty(0, dtype=np.int64)
            self._normalized_vectors = np.empty((0, 0), dtype=np.float32)
            return

        dimensions = {record.values.size for record in self._records.values()}
        if len(dimensions) != 1:
            raise ValueError("All indexed vectors must have the same dimension")

        self._item_ids = np.fromiter(self._records.keys(), dtype=np.int64)
        matrix = np.vstack(
            [record.values for record in self._records.values()]
        ).astype(np.float32, copy=False)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._normalized_vectors = np.divide(
            matrix,
            norms,
            out=np.zeros_like(matrix),
            where=norms != 0,
        )

    def _eligible_indices(self, filters: VectorFilter | None) -> np.ndarray:
        assert self._records is not None

        if not filters:
            return np.arange(len(self._item_ids))

        eligible: list[int] = []
        records = list(self._records.values())
        for index, record in enumerate(records):
            if self._metadata_matches(record.metadata, filters):
                eligible.append(index)
        return np.asarray(eligible, dtype=np.int64)

    @staticmethod
    def _metadata_matches(
        metadata: Mapping[str, object], filters: VectorFilter
    ) -> bool:
        """Support equality filters locally; managed backends may support more."""

        return all(metadata.get(key) == value for key, value in filters.items())

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: VectorFilter | None = None,
    ) -> list[SearchHit]:
        if top_k <= 0:
            return []

        self._ensure_loaded()
        assert self._records is not None
        if not self._records:
            return []

        query = self._as_vector(query_vector)
        expected_dimension = self._normalized_vectors.shape[1]
        if query.size != expected_dimension:
            raise ValueError(
                "Query vector dimension "
                f"{query.size} does not match index dimension {expected_dimension}"
            )

        query_norm = np.linalg.norm(query)
        normalized_query = (
            query / query_norm if query_norm != 0 else np.zeros_like(query)
        )
        scores = self._normalized_vectors @ normalized_query
        eligible = self._eligible_indices(filters)
        if eligible.size == 0:
            return []

        eligible_scores = scores[eligible]
        ranked_positions = np.argsort(-eligible_scores, kind="stable")[:top_k]
        ranked_indices = eligible[ranked_positions]
        records = list(self._records.values())

        return [
            SearchHit(
                item_id=int(self._item_ids[index]),
                score=float(scores[index]),
                metadata=dict(records[index].metadata),
            )
            for index in ranked_indices
        ]

    async def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return

        self._ensure_loaded()
        assert self._records is not None

        prepared = [
            VectorRecord(
                item_id=int(record.item_id),
                values=self._as_vector(record.values),
                metadata=dict(record.metadata),
            )
            for record in records
        ]
        dimensions = {record.values.size for record in prepared}
        if len(dimensions) != 1:
            raise ValueError("All upserted vectors must have the same dimension")
        if self._records:
            existing_dimension = next(iter(self._records.values())).values.size
            incoming_dimension = prepared[0].values.size
            if incoming_dimension != existing_dimension:
                raise ValueError(
                    "Upserted vector dimension "
                    f"{incoming_dimension} does not match index dimension "
                    f"{existing_dimension}"
                )

        for record in prepared:
            self._records[record.item_id] = record
        self._rebuild_index()

    async def delete(self, item_ids: Sequence[int]) -> None:
        if not item_ids:
            return

        self._ensure_loaded()
        assert self._records is not None
        for item_id in item_ids:
            self._records.pop(int(item_id), None)
        self._rebuild_index()

    async def close(self) -> None:
        return None
