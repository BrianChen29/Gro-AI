import os
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from llm_modules.catalog_grounding import build_catalog_context
from vector.in_memory_store import InMemoryVectorStore


class FakeBase(DeclarativeBase):
    pass


class FakeGroceryItemModel(FakeBase):
    __tablename__ = "integration_test_grocery_items"

    id: Mapped[int] = mapped_column(primary_key=True)


def grocery_item(
    item_id: int,
    title: str,
    *,
    price: str,
    sub_category: str,
    rating: float | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=item_id,
        title=title,
        price=Decimal(price),
        sub_category=sub_category,
        rating_value=rating,
    )


def database_result(items: list[SimpleNamespace]) -> Mock:
    result = Mock()
    result.scalars.return_value.all.return_value = items
    return result


class CatalogRetrievalIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_builds_context_through_the_complete_offline_retrieval_chain(self):
        product_names = ["oat milk", "birthday dessert"]
        embedding_provider = AsyncMock(
            return_value=[
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        store = InMemoryVectorStore(
            lambda: [
                (101, np.array([1.0, 0.0], dtype=np.float32)),
                (102, np.array([0.9, 0.43589], dtype=np.float32)),
                (201, np.array([0.0, 1.0], dtype=np.float32)),
                (301, np.array([0.7, -0.71414], dtype=np.float32)),
            ]
        )
        real_store_search = store.search
        store.search = AsyncMock(wraps=real_store_search)

        oat_milk = grocery_item(
            101,
            "Organic Oat Milk",
            price="4.25",
            sub_category="Beverages",
            rating=4.8,
        )
        almond_drink = grocery_item(
            102,
            "Unsweetened Almond Drink",
            price="3.75",
            sub_category="Beverages",
            rating=4.4,
        )
        birthday_cake = grocery_item(
            201,
            "Chocolate Birthday Cake",
            price="12.00",
            sub_category="Bakery",
            rating=None,
        )
        session = Mock()
        session.execute = AsyncMock(
            return_value=database_result([birthday_cake, almond_drink, oat_milk])
        )

        with (
            patch.dict(os.environ, {"VECTOR_SEARCH_MIN_SCORE": "0.80"}),
            patch(
                "vector.embedding_client.get_embeddings",
                embedding_provider,
            ),
            patch(
                "vector.vector_search.get_vector_store",
                return_value=store,
            ),
            patch(
                "vector.recommend_utils._get_grocery_item_model",
                return_value=FakeGroceryItemModel,
            ),
        ):
            context = await build_catalog_context(
                session,
                product_names,
                limit=3,
            )

        embedding_provider.assert_awaited_once_with(product_names)
        self.assertEqual(store.search.await_count, 2)
        self.assertTrue(
            all(call.kwargs == {"top_k": 3} for call in store.search.await_args_list)
        )
        session.execute.assert_awaited_once()
        statement = session.execute.await_args.args[0]
        self.assertEqual(
            next(iter(statement.compile().params.values())),
            [101, 102, 201],
        )
        self.assertEqual(
            context,
            [
                {
                    "title": "Organic Oat Milk",
                    "sub_category": "Beverages",
                    "price": 4.25,
                    "rating": 4.8,
                },
                {
                    "title": "Unsweetened Almond Drink",
                    "sub_category": "Beverages",
                    "price": 3.75,
                    "rating": 4.4,
                },
                {
                    "title": "Chocolate Birthday Cake",
                    "sub_category": "Bakery",
                    "price": 12.0,
                    "rating": 0.0,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
