from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from sqlalchemy import select

from vector.vector_search import search_similar_items_batch

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from db import GroceryItem


def _get_grocery_item_model():
    """Load the application model only at the database integration boundary."""

    from db import GroceryItem

    return GroceryItem


async def get_relevant_grocery_items_batch(
    session: AsyncSession,
    product_names: Sequence[str],
    limit: int = 10,
) -> list[list[GroceryItem]]:
    """Retrieve and hydrate ranked catalog matches for multiple queries."""

    names = list(product_names)
    if not names:
        return []

    scored_batches = await search_similar_items_batch(names, top_k=limit)
    if len(scored_batches) != len(names):
        raise RuntimeError(
            "Batch vector search returned a different number of result groups "
            "than requested"
        )

    ranked_id_batches = [
        [int(item_id) for item_id, _ in scored_items] for scored_items in scored_batches
    ]
    unique_item_ids = list(
        dict.fromkeys(
            item_id for ranked_ids in ranked_id_batches for item_id in ranked_ids
        )
    )
    if not unique_item_ids:
        return [[] for _ in names]

    grocery_item_model = _get_grocery_item_model()
    result = await session.execute(
        select(grocery_item_model).where(grocery_item_model.id.in_(unique_item_ids))
    )
    items_by_id = {int(item.id): item for item in result.scalars().all()}
    return [
        [items_by_id[item_id] for item_id in ranked_ids if item_id in items_by_id]
        for ranked_ids in ranked_id_batches
    ]


async def get_relevant_grocery_items(
    session: AsyncSession,
    product_name: str,
    limit: int = 10,
) -> list[GroceryItem]:
    """Return hydrated catalog matches for one product query."""

    return (
        await get_relevant_grocery_items_batch(
            session,
            [product_name],
            limit=limit,
        )
    )[0]
