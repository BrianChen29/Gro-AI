import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from vector.recommend_utils import (
    get_relevant_grocery_items,
    get_relevant_grocery_items_batch,
)


class FakeBase(DeclarativeBase):
    pass


class FakeGroceryItemModel(FakeBase):
    __tablename__ = "test_grocery_items"

    id: Mapped[int] = mapped_column(primary_key=True)


def grocery_item(item_id: int, title: str) -> SimpleNamespace:
    return SimpleNamespace(id=item_id, title=title)


def database_result(items: list[SimpleNamespace]) -> Mock:
    result = Mock()
    result.scalars.return_value.all.return_value = items
    return result


class RelevantGroceryItemsBatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_one_query_and_rebuilds_each_vector_ranking(self):
        first = grocery_item(1, "First")
        second = grocery_item(2, "Second")
        third = grocery_item(3, "Third")
        session = Mock()
        session.execute = AsyncMock(
            return_value=database_result([first, third, second])
        )
        vector_search = AsyncMock(
            return_value=[
                [(3, 0.95), (1, 0.80)],
                [(2, 0.90), (3, 0.85)],
                [(99, 0.88)],
            ]
        )
        product_names = ["oat milk", "birthday cake", "stale item"]

        with (
            patch(
                "vector.recommend_utils.search_similar_items_batch",
                vector_search,
            ),
            patch(
                "vector.recommend_utils._get_grocery_item_model",
                return_value=FakeGroceryItemModel,
            ),
        ):
            matches = await get_relevant_grocery_items_batch(
                session,
                product_names,
                limit=2,
            )

        vector_search.assert_awaited_once_with(product_names, top_k=2)
        session.execute.assert_awaited_once()
        statement = session.execute.await_args.args[0]
        self.assertEqual(
            next(iter(statement.compile().params.values())),
            [3, 1, 2, 99],
        )
        self.assertEqual(
            [[item.id for item in result] for result in matches],
            [[3, 1], [2, 3], []],
        )

    async def test_all_empty_vector_results_skip_database_query(self):
        session = Mock()
        session.execute = AsyncMock()
        vector_search = AsyncMock(return_value=[[], []])

        with patch(
            "vector.recommend_utils.search_similar_items_batch",
            vector_search,
        ):
            matches = await get_relevant_grocery_items_batch(
                session,
                ["missing one", "missing two"],
            )

        self.assertEqual(matches, [[], []])
        session.execute.assert_not_awaited()

    async def test_empty_input_skips_vector_and_database_work(self):
        session = Mock()
        session.execute = AsyncMock()
        vector_search = AsyncMock()

        with patch(
            "vector.recommend_utils.search_similar_items_batch",
            vector_search,
        ):
            matches = await get_relevant_grocery_items_batch(session, [])

        self.assertEqual(matches, [])
        vector_search.assert_not_awaited()
        session.execute.assert_not_awaited()

    async def test_rejects_missing_result_group_before_database_query(self):
        session = Mock()
        session.execute = AsyncMock()

        with patch(
            "vector.recommend_utils.search_similar_items_batch",
            AsyncMock(return_value=[[(1, 0.9)]]),
        ):
            with self.assertRaisesRegex(RuntimeError, "result groups"):
                await get_relevant_grocery_items_batch(
                    session,
                    ["first", "second"],
                )

        session.execute.assert_not_awaited()

    async def test_single_query_api_delegates_to_batch_hydration(self):
        session = Mock()
        expected = grocery_item(10, "Oat Milk")
        batch_hydration = AsyncMock(return_value=[[expected]])

        with patch(
            "vector.recommend_utils.get_relevant_grocery_items_batch",
            batch_hydration,
        ):
            matches = await get_relevant_grocery_items(
                session,
                "oat milk",
                limit=3,
            )

        self.assertEqual(matches, [expected])
        batch_hydration.assert_awaited_once_with(
            session,
            ["oat milk"],
            limit=3,
        )


if __name__ == "__main__":
    unittest.main()
