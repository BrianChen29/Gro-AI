"""Batch catalog grounding helpers used by Gro AI workflows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

from vector.recommend_utils import get_relevant_grocery_items_batch

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class CatalogMatch(Protocol):
    id: int
    title: str
    sub_category: str
    price: Any
    rating_value: float | None


def _catalog_context_item(item: CatalogMatch) -> dict[str, object]:
    return {
        "title": item.title,
        "sub_category": item.sub_category,
        "price": float(item.price),
        "rating": item.rating_value or 0.0,
    }


def _real_product(item: CatalogMatch) -> dict[str, object]:
    return {
        "id": item.id,
        "title": item.title,
        "price": float(item.price),
        "sub_category": item.sub_category,
        "rating": item.rating_value or 0.0,
    }


async def build_catalog_context(
    session: AsyncSession,
    product_names: Sequence[str],
    *,
    limit: int = 5,
) -> list[dict[str, object]]:
    """Retrieve product groups once and flatten unique titles for an LLM."""

    names = list(product_names)
    if not names:
        return []

    match_groups = await get_relevant_grocery_items_batch(
        session,
        names,
        limit=limit,
    )
    if len(match_groups) != len(names):
        raise RuntimeError(
            "Catalog retrieval returned a different number of result groups "
            "than requested"
        )
    seen_titles: set[str] = set()
    catalog_context: list[dict[str, object]] = []
    for matches in match_groups:
        for match in matches:
            if match.title in seen_titles:
                continue
            seen_titles.add(match.title)
            catalog_context.append(_catalog_context_item(match))
    return catalog_context


async def enrich_procurement_items(
    session: AsyncSession,
    plan_items: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach at most one real catalog product to each named plan item."""

    enriched_items = list(plan_items)
    searchable_items: list[tuple[dict[str, Any], str]] = []
    for item in enriched_items:
        item["match_found"] = False
        item["real_product"] = None
        raw_name = item.get("name")
        if raw_name:
            searchable_items.append((item, raw_name))

    if not searchable_items:
        return enriched_items

    match_groups = await get_relevant_grocery_items_batch(
        session,
        [raw_name for _, raw_name in searchable_items],
        limit=1,
    )
    if len(match_groups) != len(searchable_items):
        raise RuntimeError(
            "Catalog retrieval returned a different number of result groups "
            "than requested"
        )
    for (item, raw_name), matches in zip(
        searchable_items,
        match_groups,
        strict=True,
    ):
        if not matches:
            print(f"No match found for '{raw_name}'")
            continue
        item["match_found"] = True
        item["real_product"] = _real_product(matches[0])
        print(f"Matched '{raw_name}' -> '{matches[0].title}'")

    return enriched_items
