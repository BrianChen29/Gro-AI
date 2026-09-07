"""OpenAI embedding client shared by indexing and query-time retrieval."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


def get_embedding_model() -> str:
    """Return the one embedding model shared by indexing and querying."""

    return (
        os.getenv("OPENAI_EMBEDDING_MODEL", "").strip()
        or DEFAULT_EMBEDDING_MODEL
    )


def _new_client() -> Any:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required to generate embeddings"
        ) from exc
    return AsyncOpenAI()


async def get_embeddings(
    texts: Sequence[str],
    model: str | None = None,
) -> list[list[float]]:
    """Embed a batch of texts and preserve the caller's input order."""

    inputs = list(texts)
    if not inputs:
        return []
    if len(inputs) > 2048:
        raise ValueError(
            "An embedding batch cannot contain more than 2048 inputs"
        )
    if any(not isinstance(text, str) or not text.strip() for text in inputs):
        raise ValueError("Embedding inputs must be non-empty strings")

    selected_model = (model or get_embedding_model()).strip()
    if not selected_model:
        raise ValueError("Embedding model must not be empty")

    client = _new_client()
    response = await client.embeddings.create(
        model=selected_model,
        input=inputs,
    )

    vectors_by_index: dict[int, list[float]] = {}
    for result in response.data:
        index = int(result.index)
        if index < 0 or index >= len(inputs) or index in vectors_by_index:
            raise RuntimeError("OpenAI returned invalid embedding result indexes")
        vector = list(result.embedding)
        if not vector:
            raise RuntimeError(
                f"OpenAI returned an empty embedding at index {index}"
            )
        vectors_by_index[index] = vector

    if len(vectors_by_index) != len(inputs):
        raise RuntimeError(
            "OpenAI returned a different number of embeddings than requested"
        )
    return [vectors_by_index[index] for index in range(len(inputs))]


async def get_embedding(
    text: str,
    model: str | None = None,
) -> list[float]:
    """Embed one text while keeping the existing single-item interface."""

    return (await get_embeddings([text], model=model))[0]
