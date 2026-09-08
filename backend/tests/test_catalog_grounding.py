import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from llm_modules.catalog_grounding import (
    build_catalog_context,
    enrich_procurement_items,
)


def catalog_item(
    item_id: int,
    title: str,
    *,
    price: str = "3.50",
    sub_category: str = "Pantry",
    rating: float | None = 4.5,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=item_id,
        title=title,
        price=Decimal(price),
        sub_category=sub_category,
        rating_value=rating,
    )


class CatalogContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetches_one_batch_and_preserves_first_unique_title(self):
        session = Mock()
        first_oat_milk = catalog_item(
            10,
            "Oat Milk",
            price="4.25",
            sub_category="Beverages",
            rating=4.8,
        )
        duplicate_oat_milk = catalog_item(
            30,
            "Oat Milk",
            price="9.99",
            sub_category="Duplicate",
            rating=1.0,
        )
        cake = catalog_item(
            20,
            "Chocolate Cake",
            price="12.00",
            sub_category="Bakery",
            rating=None,
        )
        batch_retrieval = AsyncMock(
            return_value=[
                [first_oat_milk, cake],
                [duplicate_oat_milk],
            ]
        )
        product_names = ["oat milk", "birthday dessert"]

        with patch(
            "llm_modules.catalog_grounding.get_relevant_grocery_items_batch",
            batch_retrieval,
        ):
            context = await build_catalog_context(
                session,
                product_names,
                limit=5,
            )

        batch_retrieval.assert_awaited_once_with(
            session,
            product_names,
            limit=5,
        )
        self.assertEqual(
            context,
            [
                {
                    "title": "Oat Milk",
                    "sub_category": "Beverages",
                    "price": 4.25,
                    "rating": 4.8,
                },
                {
                    "title": "Chocolate Cake",
                    "sub_category": "Bakery",
                    "price": 12.0,
                    "rating": 0.0,
                },
            ],
        )

    async def test_empty_catalog_context_skips_batch_retrieval(self):
        batch_retrieval = AsyncMock()

        with patch(
            "llm_modules.catalog_grounding.get_relevant_grocery_items_batch",
            batch_retrieval,
        ):
            context = await build_catalog_context(Mock(), [])

        self.assertEqual(context, [])
        batch_retrieval.assert_not_awaited()

    async def test_catalog_context_rejects_missing_result_group(self):
        with patch(
            "llm_modules.catalog_grounding.get_relevant_grocery_items_batch",
            AsyncMock(return_value=[[catalog_item(10, "Milk")]]),
        ):
            with self.assertRaisesRegex(RuntimeError, "result groups"):
                await build_catalog_context(
                    Mock(),
                    ["milk", "cake"],
                )


class ProcurementItemEnrichmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_batches_only_named_items_and_keeps_alignment(self):
        session = Mock()
        milk = catalog_item(
            10,
            "Whole Milk",
            price="3.20",
            sub_category="Dairy",
            rating=4.2,
        )
        cake = catalog_item(
            20,
            "Birthday Cake",
            price="15.00",
            sub_category="Bakery",
            rating=None,
        )
        plan_items = [
            {"name": "milk", "quantity": 2},
            {"quantity": 1},
            {"name": "cake", "quantity": 1},
            {"name": "not in catalog", "quantity": 4},
        ]
        batch_retrieval = AsyncMock(return_value=[[milk], [cake], []])

        with (
            patch(
                "llm_modules.catalog_grounding.get_relevant_grocery_items_batch",
                batch_retrieval,
            ),
            patch("builtins.print") as print_message,
        ):
            enriched = await enrich_procurement_items(session, plan_items)

        batch_retrieval.assert_awaited_once_with(
            session,
            ["milk", "cake", "not in catalog"],
            limit=1,
        )
        self.assertTrue(enriched[0]["match_found"])
        self.assertEqual(
            enriched[0]["real_product"],
            {
                "id": 10,
                "title": "Whole Milk",
                "price": 3.2,
                "sub_category": "Dairy",
                "rating": 4.2,
            },
        )
        self.assertFalse(enriched[1]["match_found"])
        self.assertIsNone(enriched[1]["real_product"])
        self.assertEqual(enriched[1]["quantity"], 1)
        self.assertTrue(enriched[2]["match_found"])
        self.assertEqual(enriched[2]["real_product"]["id"], 20)
        self.assertFalse(enriched[3]["match_found"])
        self.assertIsNone(enriched[3]["real_product"])
        self.assertEqual(print_message.call_count, 3)

    async def test_items_without_names_skip_batch_retrieval(self):
        batch_retrieval = AsyncMock()
        plan_items = [{"quantity": 1}, {"name": "", "quantity": 2}]

        with patch(
            "llm_modules.catalog_grounding.get_relevant_grocery_items_batch",
            batch_retrieval,
        ):
            enriched = await enrich_procurement_items(Mock(), plan_items)

        batch_retrieval.assert_not_awaited()
        self.assertEqual(
            enriched,
            [
                {"quantity": 1, "match_found": False, "real_product": None},
                {
                    "name": "",
                    "quantity": 2,
                    "match_found": False,
                    "real_product": None,
                },
            ],
        )

    async def test_result_group_mismatch_is_not_silently_truncated(self):
        with patch(
            "llm_modules.catalog_grounding.get_relevant_grocery_items_batch",
            AsyncMock(return_value=[[catalog_item(10, "Milk")]]),
        ):
            with self.assertRaisesRegex(RuntimeError, "result groups"):
                await enrich_procurement_items(
                    Mock(),
                    [{"name": "milk"}, {"name": "cake"}],
                )


if __name__ == "__main__":
    unittest.main()
