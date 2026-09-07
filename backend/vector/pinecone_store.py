"""Pinecone implementation of the catalog vector-store contract."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from vector.store import SearchHit, VectorFilter, VectorRecord


class PineconeVectorStore:
    """Use Pinecone's async data-plane client for catalog vector operations."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        index_host: str | None = None,
        namespace: str = "catalog",
        batch_size: int = 50,
        index: Any | None = None,
    ) -> None:
        api_key = api_key.strip() if api_key else None
        index_host = index_host.strip() if index_host else None

        if batch_size <= 0:
            raise ValueError("PINECONE_BATCH_SIZE must be greater than zero")
        if index is None and (not api_key or not index_host):
            raise ValueError(
                "PINECONE_API_KEY and PINECONE_INDEX_HOST are required "
                "when VECTOR_STORE_BACKEND=pinecone"
            )

        self._api_key = api_key
        self._index_host = index_host
        self._namespace = namespace
        self._batch_size = batch_size
        self._index = index

    @classmethod
    def from_env(cls) -> "PineconeVectorStore":
        raw_batch_size = os.getenv("PINECONE_BATCH_SIZE", "50")
        try:
            batch_size = int(raw_batch_size)
        except ValueError as exc:
            raise ValueError(
                "PINECONE_BATCH_SIZE must be a positive integer"
            ) from exc

        return cls(
            api_key=os.getenv("PINECONE_API_KEY"),
            index_host=os.getenv("PINECONE_INDEX_HOST"),
            namespace=os.getenv("PINECONE_NAMESPACE", "catalog"),
            batch_size=batch_size,
        )

    def _get_index(self) -> Any:
        if self._index is not None:
            return self._index

        try:
            from pinecone import AsyncIndex
        except ImportError as exc:
            raise RuntimeError(
                "The pinecone package is required for the Pinecone backend"
            ) from exc

        self._index = AsyncIndex(
            host=self._index_host,
            api_key=self._api_key,
        )
        return self._index

    @staticmethod
    def _as_vector(values: np.ndarray) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise ValueError("Vector values must not be empty")
        if not np.all(np.isfinite(vector)):
            raise ValueError("Vector values must contain only finite numbers")
        return vector

    @staticmethod
    def _response_value(response: Any, key: str, default: Any = None) -> Any:
        if isinstance(response, Mapping):
            return response.get(key, default)
        return getattr(response, key, default)

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: VectorFilter | None = None,
    ) -> list[SearchHit]:
        if top_k <= 0:
            return []

        request: dict[str, Any] = {
            "namespace": self._namespace,
            "vector": self._as_vector(query_vector).tolist(),
            "top_k": top_k,
            "include_metadata": True,
            "include_values": False,
        }
        if filters:
            request["filter"] = dict(filters)

        response = await self._get_index().query(**request)
        matches = self._response_value(response, "matches", ()) or ()
        hits: list[SearchHit] = []
        for match in matches:
            raw_id = self._response_value(match, "id")
            try:
                item_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Pinecone returned a non-catalog vector ID: {raw_id!r}"
                ) from exc

            metadata = self._response_value(match, "metadata", {}) or {}
            hits.append(
                SearchHit(
                    item_id=item_id,
                    score=float(self._response_value(match, "score", 0.0)),
                    metadata=dict(metadata),
                )
            )
        return hits

    async def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return

        vectors = [
            {
                "id": str(record.item_id),
                "values": self._as_vector(record.values).tolist(),
                "metadata": dict(record.metadata),
            }
            for record in records
        ]
        response = await self._get_index().upsert(
            vectors=vectors,
            namespace=self._namespace,
            batch_size=self._batch_size,
            show_progress=False,
        )
        if self._response_value(response, "has_errors", False):
            errors = self._response_value(response, "errors", "unknown error")
            raise RuntimeError(f"Pinecone upsert had partial failures: {errors}")

    async def delete(self, item_ids: Sequence[int]) -> None:
        if not item_ids:
            return

        await self._get_index().delete(
            ids=[str(item_id) for item_id in item_ids],
            namespace=self._namespace,
        )

    async def close(self) -> None:
        if self._index is None:
            return

        await self._index.close()
        self._index = None
