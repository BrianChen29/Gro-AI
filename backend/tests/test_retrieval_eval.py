import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from vector.retrieval_eval import (
    RetrievalEvaluationCase,
    _build_parser,
    _run_cli,
    evaluate_retrieval,
    load_evaluation_dataset,
)
from vector.store import SearchHit


def write_dataset(path: Path, cases: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "description": "Catalog retrieval regression cases",
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )


class RetrievalEvaluationDatasetTests(unittest.TestCase):
    def test_loads_versioned_cases_and_optional_filters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            write_dataset(
                path,
                [
                    {
                        "case_id": "oat-milk",
                        "query": "  unsweetened oat milk  ",
                        "relevant_item_ids": [10, 11],
                        "filters": {"sub_category": "Beverages & Water"},
                    }
                ],
            )

            dataset = load_evaluation_dataset(path)

        self.assertEqual(dataset.schema_version, 1)
        self.assertEqual(dataset.description, "Catalog retrieval regression cases")
        self.assertEqual(dataset.cases[0].case_id, "oat-milk")
        self.assertEqual(dataset.cases[0].query, "unsweetened oat milk")
        self.assertEqual(dataset.cases[0].relevant_item_ids, (10, 11))
        self.assertEqual(
            dataset.cases[0].filters,
            {"sub_category": "Beverages & Water"},
        )

    def test_rejects_unknown_schema_and_duplicate_case_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            wrong_schema_path = directory_path / "wrong-schema.json"
            wrong_schema_path.write_text(
                json.dumps({"schema_version": 2, "cases": [{}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_evaluation_dataset(wrong_schema_path)

            duplicate_path = directory_path / "duplicate.json"
            duplicate_case = {
                "case_id": "same-id",
                "query": "milk",
                "relevant_item_ids": [10],
            }
            write_dataset(duplicate_path, [duplicate_case, duplicate_case])
            with self.assertRaisesRegex(ValueError, "duplicate case IDs"):
                load_evaluation_dataset(duplicate_path)

    def test_rejects_invalid_relevance_labels_and_unknown_fields(self):
        invalid_cases = [
            {
                "case_id": "empty-labels",
                "query": "milk",
                "relevant_item_ids": [],
            },
            {
                "case_id": "duplicate-labels",
                "query": "milk",
                "relevant_item_ids": [10, 10],
            },
            {
                "case_id": "boolean-label",
                "query": "milk",
                "relevant_item_ids": [True],
            },
            {
                "case_id": "typo",
                "query": "milk",
                "relevant_item_ids": [10],
                "relevent_item_ids": [10],
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            for index, invalid_case in enumerate(invalid_cases):
                with self.subTest(index=index):
                    path = Path(directory) / f"invalid-{index}.json"
                    write_dataset(path, [invalid_case])
                    with self.assertRaises(ValueError):
                        load_evaluation_dataset(path)


class RetrievalMetricTests(unittest.IsolatedAsyncioTestCase):
    async def test_computes_metrics_from_exact_ranked_results(self):
        cases = (
            RetrievalEvaluationCase(
                case_id="case-a",
                query="oat milk",
                relevant_item_ids=(10, 20),
                filters={"sub_category": "Beverages & Water"},
            ),
            RetrievalEvaluationCase(
                case_id="case-b",
                query="birthday cake",
                relevant_item_ids=(40,),
            ),
        )
        embed_batch = AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0]])
        store = Mock()
        store.search = AsyncMock(
            side_effect=[
                [
                    SearchHit(item_id=30, score=0.95),
                    SearchHit(
                        item_id=20,
                        score=0.90,
                        metadata={"title": "Oat Milk"},
                    ),
                    SearchHit(item_id=10, score=0.80),
                ],
                [
                    SearchHit(item_id=50, score=0.75),
                    SearchHit(item_id=60, score=0.60),
                ],
            ]
        )

        report = await evaluate_retrieval(
            cases,
            store,
            embed_batch,
            top_k=3,
        )

        embed_batch.assert_awaited_once_with(["oat milk", "birthday cake"])
        first_call, second_call = store.search.await_args_list
        np.testing.assert_array_equal(first_call.args[0], [1.0, 0.0])
        self.assertEqual(first_call.kwargs["top_k"], 3)
        self.assertEqual(
            first_call.kwargs["filters"],
            {"sub_category": "Beverages & Water"},
        )
        np.testing.assert_array_equal(second_call.args[0], [0.0, 1.0])
        self.assertIsNone(second_call.kwargs["filters"])

        first_result = report.case_results[0]
        self.assertTrue(first_result.hit_at_k)
        self.assertAlmostEqual(first_result.precision_at_k, 2 / 3)
        self.assertEqual(first_result.recall_at_k, 1.0)
        self.assertEqual(first_result.reciprocal_rank_at_k, 0.5)
        self.assertEqual(
            [hit.relevant for hit in first_result.hits],
            [False, True, True],
        )
        self.assertEqual(first_result.hits[1].metadata["title"], "Oat Milk")

        self.assertEqual(report.hit_rate_at_k, 0.5)
        self.assertAlmostEqual(report.mean_precision_at_k, 1 / 3)
        self.assertEqual(report.mean_recall_at_k, 0.5)
        self.assertEqual(report.mean_reciprocal_rank_at_k, 0.25)

    async def test_invalid_embedding_batch_stops_before_search(self):
        case = RetrievalEvaluationCase(
            case_id="milk",
            query="milk",
            relevant_item_ids=(10,),
        )
        store = Mock()
        store.search = AsyncMock()

        with self.assertRaisesRegex(RuntimeError, "different number"):
            await evaluate_retrieval(
                [case],
                store,
                AsyncMock(return_value=[]),
            )

        store.search.assert_not_awaited()

    async def test_duplicate_result_ids_are_rejected(self):
        case = RetrievalEvaluationCase(
            case_id="milk",
            query="milk",
            relevant_item_ids=(10,),
        )
        store = Mock()
        store.search = AsyncMock(
            return_value=[
                SearchHit(item_id=10, score=0.9),
                SearchHit(item_id=10, score=0.8),
            ]
        )

        with self.assertRaisesRegex(ValueError, "duplicate item IDs"):
            await evaluate_retrieval(
                [case],
                store,
                AsyncMock(return_value=[[1.0, 0.0]]),
            )

    async def test_applies_inclusive_score_threshold_before_metrics(self):
        case = RetrievalEvaluationCase(
            case_id="milk",
            query="milk",
            relevant_item_ids=(10, 20),
        )
        store = Mock()
        store.search = AsyncMock(
            return_value=[
                SearchHit(item_id=30, score=0.90),
                SearchHit(item_id=10, score=0.80),
                SearchHit(item_id=20, score=0.79),
            ]
        )

        report = await evaluate_retrieval(
            [case],
            store,
            AsyncMock(return_value=[[1.0, 0.0]]),
            top_k=3,
            min_score=0.80,
        )

        self.assertEqual(report.applied_min_score, 0.80)
        self.assertEqual(
            [hit.item_id for hit in report.case_results[0].hits],
            [30, 10],
        )
        self.assertAlmostEqual(report.mean_precision_at_k, 1 / 3)
        self.assertEqual(report.mean_recall_at_k, 0.5)
        self.assertEqual(report.mean_reciprocal_rank_at_k, 0.5)

    async def test_rejects_hits_that_are_not_in_descending_score_order(self):
        case = RetrievalEvaluationCase(
            case_id="milk",
            query="milk",
            relevant_item_ids=(10,),
        )
        store = Mock()
        store.search = AsyncMock(
            return_value=[
                SearchHit(item_id=10, score=0.80),
                SearchHit(item_id=20, score=0.90),
            ]
        )

        with self.assertRaisesRegex(ValueError, "out of score order"):
            await evaluate_retrieval(
                [case],
                store,
                AsyncMock(return_value=[[1.0, 0.0]]),
            )


class RetrievalEvaluationCliTests(unittest.IsolatedAsyncioTestCase):
    def test_parser_defaults_to_top_five(self):
        args = _build_parser().parse_args(["--cases", "cases.json"])

        self.assertEqual(args.top_k, 5)
        self.assertFalse(args.validate_only)
        self.assertIsNone(args.min_score)

    def test_parser_validates_cosine_score_threshold(self):
        args = _build_parser().parse_args(
            ["--cases", "cases.json", "--min-score", "0.72"]
        )

        self.assertEqual(args.min_score, 0.72)
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                _build_parser().parse_args(
                    ["--cases", "cases.json", "--min-score", "1.1"]
                )

    async def test_validate_only_does_not_load_embedding_or_vector_store(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            write_dataset(
                path,
                [
                    {
                        "case_id": "milk",
                        "query": "milk",
                        "relevant_item_ids": [10],
                    }
                ],
            )
            output = io.StringIO()
            args = SimpleNamespace(
                cases=path,
                top_k=5,
                validate_only=True,
            )

            with (
                patch("vector.factory.get_vector_store") as get_store,
                patch("vector.embedding_client.get_embeddings") as get_embeddings,
                redirect_stdout(output),
            ):
                await _run_cli(args)

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["mode"], "validate-only")
        self.assertEqual(payload["case_ids"], ["milk"])
        get_store.assert_not_called()
        get_embeddings.assert_not_called()

    async def test_memory_evaluation_requires_a_nonempty_embedding_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            cases_path = directory_path / "cases.json"
            missing_cache_path = directory_path / "missing.sqlite"
            write_dataset(
                cases_path,
                [
                    {
                        "case_id": "milk",
                        "query": "milk",
                        "relevant_item_ids": [10],
                    }
                ],
            )
            args = SimpleNamespace(
                cases=cases_path,
                top_k=5,
                validate_only=False,
            )

            with (
                patch(
                    "vector.factory.uses_local_embedding_cache",
                    return_value=True,
                ),
                patch(
                    "vector.vector_cache.get_embeddings_db_path",
                    return_value=missing_cache_path,
                ),
                patch("vector.factory.get_vector_store") as get_store,
            ):
                with self.assertRaisesRegex(FileNotFoundError, "not found"):
                    await _run_cli(args)

        get_store.assert_not_called()

    async def test_cli_outputs_backend_model_metrics_and_case_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            write_dataset(
                path,
                [
                    {
                        "case_id": "milk",
                        "query": "milk",
                        "relevant_item_ids": [10],
                    }
                ],
            )
            store = Mock()

            async def noisy_search(*args, **kwargs):
                print("backend status message")
                return [SearchHit(item_id=10, score=0.91)]

            store.search = AsyncMock(side_effect=noisy_search)
            close_store = AsyncMock()
            output = io.StringIO()
            errors = io.StringIO()
            args = SimpleNamespace(
                cases=path,
                top_k=1,
                validate_only=False,
            )

            with (
                patch("vector.factory.get_vector_store", return_value=store),
                patch("vector.factory.close_vector_store", close_store),
                patch(
                    "vector.factory.vector_store_backend_name",
                    return_value="pinecone",
                ),
                patch(
                    "vector.factory.uses_local_embedding_cache",
                    return_value=False,
                ),
                patch(
                    "vector.embedding_client.get_embedding_model",
                    return_value="test-embedding-model",
                ),
                patch(
                    "vector.embedding_client.get_embeddings",
                    AsyncMock(return_value=[[1.0, 0.0]]),
                ),
                redirect_stdout(output),
                redirect_stderr(errors),
            ):
                await _run_cli(args)

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["mode"], "evaluate")
        self.assertEqual(payload["vector_store_backend"], "pinecone")
        self.assertEqual(payload["embedding_model"], "test-embedding-model")
        self.assertEqual(payload["metrics"]["hit_rate_at_k"], 1.0)
        self.assertEqual(payload["cases"][0]["hits"][0]["score"], 0.91)
        self.assertEqual(
            payload["score_threshold_analysis"]["mode"],
            "calibration",
        )
        self.assertEqual(
            payload["score_threshold_analysis"]["status"],
            "unavailable",
        )
        self.assertIn("backend status message", errors.getvalue())
        close_store.assert_awaited_once_with()

    async def test_cli_applies_fixed_threshold_in_validation_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "held-out-cases.json"
            write_dataset(
                path,
                [
                    {
                        "case_id": "milk",
                        "query": "milk",
                        "relevant_item_ids": [10, 20],
                    }
                ],
            )
            store = Mock()
            store.search = AsyncMock(
                return_value=[
                    SearchHit(item_id=10, score=0.91),
                    SearchHit(item_id=30, score=0.80),
                    SearchHit(item_id=20, score=0.70),
                ]
            )
            output = io.StringIO()
            args = SimpleNamespace(
                cases=path,
                top_k=3,
                min_score=0.85,
                validate_only=False,
            )

            with (
                patch("vector.factory.get_vector_store", return_value=store),
                patch("vector.factory.close_vector_store", AsyncMock()),
                patch(
                    "vector.factory.vector_store_backend_name",
                    return_value="pinecone",
                ),
                patch(
                    "vector.factory.uses_local_embedding_cache",
                    return_value=False,
                ),
                patch(
                    "vector.embedding_client.get_embedding_model",
                    return_value="test-embedding-model",
                ),
                patch(
                    "vector.embedding_client.get_embeddings",
                    AsyncMock(return_value=[[1.0, 0.0]]),
                ),
                redirect_stdout(output),
            ):
                await _run_cli(args)

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["applied_min_score"], 0.85)
        self.assertEqual(
            [hit["item_id"] for hit in payload["cases"][0]["hits"]],
            [10],
        )
        analysis = payload["score_threshold_analysis"]
        self.assertEqual(analysis["mode"], "validation")
        self.assertEqual(analysis["metrics"]["true_positive_count"], 1)
        self.assertEqual(analysis["metrics"]["false_negative_count"], 1)


if __name__ == "__main__":
    unittest.main()
