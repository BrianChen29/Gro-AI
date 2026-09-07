"""Evaluate catalog retrieval against human-labeled relevance cases."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections.abc import Awaitable, Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from vector.search_policy import filter_search_hits, validate_min_score
from vector.store import MetadataValue, SearchHit, VectorFilter, VectorStore
from vector.threshold_calibration import (
    calculate_score_threshold_metrics,
    calibrate_score_threshold,
)


EVALUATION_SCHEMA_VERSION = 1
EmbeddingBatchFunction = Callable[
    [Sequence[str]],
    Awaitable[Sequence[Sequence[float]]],
]


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCase:
    """One query and the catalog IDs judged relevant for it."""

    case_id: str
    query: str
    relevant_item_ids: tuple[int, ...]
    filters: VectorFilter = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationDataset:
    """Versioned collection of retrieval relevance judgments."""

    cases: tuple[RetrievalEvaluationCase, ...]
    description: str | None = None
    schema_version: int = EVALUATION_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class RankedEvaluationHit:
    rank: int
    item_id: int
    score: float
    relevant: bool
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "item_id": self.item_id,
            "score": self.score,
            "relevant": self.relevant,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RetrievalCaseResult:
    case_id: str
    query: str
    relevant_item_ids: tuple[int, ...]
    filters: VectorFilter
    hits: tuple[RankedEvaluationHit, ...]
    hit_at_k: bool
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank_at_k: float

    def as_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "relevant_item_ids": list(self.relevant_item_ids),
            "filters": dict(self.filters),
            "hit_at_k": self.hit_at_k,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "reciprocal_rank_at_k": self.reciprocal_rank_at_k,
            "hits": [hit.as_dict() for hit in self.hits],
        }


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationReport:
    top_k: int
    applied_min_score: float | None
    case_results: tuple[RetrievalCaseResult, ...]
    hit_rate_at_k: float
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank_at_k: float

    def as_dict(self) -> dict[str, object]:
        return {
            "case_count": len(self.case_results),
            "top_k": self.top_k,
            "applied_min_score": self.applied_min_score,
            "metrics": {
                "hit_rate_at_k": self.hit_rate_at_k,
                "mean_precision_at_k": self.mean_precision_at_k,
                "mean_recall_at_k": self.mean_recall_at_k,
                "mean_reciprocal_rank_at_k": self.mean_reciprocal_rank_at_k,
            },
            "cases": [result.as_dict() for result in self.case_results],
        }


def _required_non_empty_string(
    raw_value: object,
    *,
    field_name: str,
    case_label: str,
) -> str:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"{case_label} must have a non-empty {field_name}")
    return raw_value.strip()


def _parse_relevant_item_ids(
    raw_ids: object,
    *,
    case_label: str,
) -> tuple[int, ...]:
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError(f"{case_label} relevant_item_ids must be a non-empty list")

    item_ids: list[int] = []
    seen: set[int] = set()
    for raw_item_id in raw_ids:
        if (
            isinstance(raw_item_id, bool)
            or not isinstance(raw_item_id, int)
            or raw_item_id <= 0
        ):
            raise ValueError(
                f"{case_label} relevant_item_ids must contain positive integers"
            )
        if raw_item_id in seen:
            raise ValueError(
                f"{case_label} has duplicate relevant item ID {raw_item_id}"
            )
        item_ids.append(raw_item_id)
        seen.add(raw_item_id)
    return tuple(item_ids)


def _parse_case(raw_case: object, *, index: int) -> RetrievalEvaluationCase:
    case_label = f"Evaluation case at index {index}"
    if not isinstance(raw_case, dict):
        raise ValueError(f"{case_label} must be a JSON object")

    allowed_fields = {"case_id", "query", "relevant_item_ids", "filters"}
    unexpected_fields = sorted(set(raw_case).difference(allowed_fields))
    if unexpected_fields:
        raise ValueError(f"{case_label} has unsupported fields: {unexpected_fields}")

    case_id = _required_non_empty_string(
        raw_case.get("case_id"),
        field_name="case_id",
        case_label=case_label,
    )
    query = _required_non_empty_string(
        raw_case.get("query"),
        field_name="query",
        case_label=f"Evaluation case {case_id!r}",
    )
    relevant_item_ids = _parse_relevant_item_ids(
        raw_case.get("relevant_item_ids"),
        case_label=f"Evaluation case {case_id!r}",
    )

    raw_filters = raw_case.get("filters", {})
    if not isinstance(raw_filters, dict):
        raise ValueError(f"Evaluation case {case_id!r} filters must be an object")
    if any(not isinstance(key, str) or not key for key in raw_filters):
        raise ValueError(
            f"Evaluation case {case_id!r} filter keys must be non-empty strings"
        )

    return RetrievalEvaluationCase(
        case_id=case_id,
        query=query,
        relevant_item_ids=relevant_item_ids,
        filters=dict(raw_filters),
    )


def load_evaluation_dataset(
    path: str | Path,
) -> RetrievalEvaluationDataset:
    """Load and strictly validate a versioned JSON evaluation dataset."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Retrieval evaluation dataset not found at {dataset_path}"
        )
    try:
        raw_dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in retrieval evaluation dataset {dataset_path}: {exc}"
        ) from exc

    if not isinstance(raw_dataset, dict):
        raise ValueError("Retrieval evaluation dataset must be a JSON object")
    allowed_fields = {"schema_version", "description", "cases"}
    unexpected_fields = sorted(set(raw_dataset).difference(allowed_fields))
    if unexpected_fields:
        raise ValueError(
            "Retrieval evaluation dataset has unsupported fields: "
            f"{unexpected_fields}"
        )

    schema_version = raw_dataset.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != EVALUATION_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported retrieval evaluation schema_version "
            f"{schema_version!r}; expected {EVALUATION_SCHEMA_VERSION}"
        )

    raw_description = raw_dataset.get("description")
    if raw_description is not None and (
        not isinstance(raw_description, str) or not raw_description.strip()
    ):
        raise ValueError("Evaluation dataset description must be a non-empty string")
    description = raw_description.strip() if raw_description else None

    raw_cases = raw_dataset.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Evaluation dataset cases must be a non-empty list")
    cases = tuple(
        _parse_case(raw_case, index=index) for index, raw_case in enumerate(raw_cases)
    )

    case_ids = [case.case_id for case in cases]
    duplicate_case_ids = sorted(
        case_id for case_id in set(case_ids) if case_ids.count(case_id) > 1
    )
    if duplicate_case_ids:
        raise ValueError(
            f"Evaluation dataset has duplicate case IDs: {duplicate_case_ids}"
        )

    return RetrievalEvaluationDataset(
        cases=cases,
        description=description,
        schema_version=schema_version,
    )


def _validate_query_vectors(
    raw_vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
) -> tuple[np.ndarray, ...]:
    if len(raw_vectors) != expected_count:
        raise RuntimeError(
            "Embedding provider returned a different number of query vectors "
            "than requested"
        )

    vectors: list[np.ndarray] = []
    dimensions: set[int] = set()
    for index, raw_vector in enumerate(raw_vectors):
        try:
            vector = np.asarray(raw_vector, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid query embedding at index {index}") from exc
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError(f"Invalid query embedding at index {index}")
        vectors.append(vector)
        dimensions.add(vector.size)

    if len(dimensions) > 1:
        raise ValueError(
            f"Query embeddings have inconsistent dimensions: {sorted(dimensions)}"
        )
    return tuple(vectors)


def _evaluate_case_hits(
    case: RetrievalEvaluationCase,
    raw_hits: Sequence[SearchHit],
    *,
    top_k: int,
    min_score: float | None,
) -> RetrievalCaseResult:
    hits = list(raw_hits[:top_k])
    hit_ids = [int(hit.item_id) for hit in hits]
    if len(hit_ids) != len(set(hit_ids)):
        raise ValueError(
            f"Vector store returned duplicate item IDs for case {case.case_id!r}"
        )

    scores: list[float] = []
    for hit in hits:
        score = float(hit.score)
        if not math.isfinite(score):
            raise ValueError(
                f"Vector store returned a non-finite score for item {hit.item_id}"
            )
        scores.append(score)
    if any(score > previous for previous, score in zip(scores, scores[1:])):
        raise ValueError(
            f"Vector store returned hits out of score order for case {case.case_id!r}"
        )

    hits = filter_search_hits(hits, min_score)
    relevant_ids = set(case.relevant_item_ids)
    ranked_hits: list[RankedEvaluationHit] = []
    relevant_ranks: list[int] = []
    for rank, hit in enumerate(hits, start=1):
        score = float(hit.score)
        is_relevant = int(hit.item_id) in relevant_ids
        if is_relevant:
            relevant_ranks.append(rank)
        ranked_hits.append(
            RankedEvaluationHit(
                rank=rank,
                item_id=int(hit.item_id),
                score=score,
                relevant=is_relevant,
                metadata=dict(hit.metadata),
            )
        )

    relevant_hit_count = len(relevant_ranks)
    return RetrievalCaseResult(
        case_id=case.case_id,
        query=case.query,
        relevant_item_ids=case.relevant_item_ids,
        filters=dict(case.filters),
        hits=tuple(ranked_hits),
        hit_at_k=bool(relevant_ranks),
        precision_at_k=relevant_hit_count / top_k,
        recall_at_k=relevant_hit_count / len(relevant_ids),
        reciprocal_rank_at_k=(1.0 / relevant_ranks[0] if relevant_ranks else 0.0),
    )


async def evaluate_retrieval(
    cases: Sequence[RetrievalEvaluationCase],
    store: VectorStore,
    embed_batch: EmbeddingBatchFunction,
    *,
    top_k: int = 5,
    min_score: float | None = None,
) -> RetrievalEvaluationReport:
    """Embed all queries once, search each case, and aggregate IR metrics."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if not cases:
        raise ValueError("At least one retrieval evaluation case is required")
    applied_min_score = validate_min_score(min_score) if min_score is not None else None

    raw_vectors = list(await embed_batch([case.query for case in cases]))
    query_vectors = _validate_query_vectors(
        raw_vectors,
        expected_count=len(cases),
    )

    case_results: list[RetrievalCaseResult] = []
    for case, query_vector in zip(cases, query_vectors, strict=True):
        hits = await store.search(
            query_vector,
            top_k=top_k,
            filters=(dict(case.filters) if case.filters else None),
        )
        case_results.append(
            _evaluate_case_hits(
                case,
                hits,
                top_k=top_k,
                min_score=applied_min_score,
            )
        )

    case_count = len(case_results)
    return RetrievalEvaluationReport(
        top_k=top_k,
        applied_min_score=applied_min_score,
        case_results=tuple(case_results),
        hit_rate_at_k=(sum(result.hit_at_k for result in case_results) / case_count),
        mean_precision_at_k=(
            sum(result.precision_at_k for result in case_results) / case_count
        ),
        mean_recall_at_k=(
            sum(result.recall_at_k for result in case_results) / case_count
        ),
        mean_reciprocal_rank_at_k=(
            sum(result.reciprocal_rank_at_k for result in case_results) / case_count
        ),
    )


async def _run_cli(args: argparse.Namespace) -> None:
    dataset = load_evaluation_dataset(args.cases)

    if args.validate_only:
        output = {
            "mode": "validate-only",
            "schema_version": dataset.schema_version,
            "dataset": str(args.cases),
            "description": dataset.description,
            "case_count": len(dataset.cases),
            "case_ids": [case.case_id for case in dataset.cases],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    from vector.embedding_client import get_embedding_model, get_embeddings
    from vector.factory import (
        close_vector_store,
        get_vector_store,
        uses_local_embedding_cache,
        vector_store_backend_name,
    )
    from vector.vector_cache import (
        get_embeddings_db_path,
        read_cached_embeddings,
    )

    backend_name = vector_store_backend_name()
    if uses_local_embedding_cache():
        embeddings_path = get_embeddings_db_path()
        cached_embeddings = read_cached_embeddings(embeddings_path)
        if not cached_embeddings:
            raise RuntimeError(
                f"Embedding cache at {embeddings_path} contains no vectors"
            )

    # Keep stdout machine-readable even if a selected backend prints status
    # messages while it initializes or searches.
    with redirect_stdout(sys.stderr):
        store = get_vector_store()
        try:
            report = await evaluate_retrieval(
                dataset.cases,
                store,
                get_embeddings,
                top_k=args.top_k,
                min_score=getattr(args, "min_score", None),
            )
        finally:
            await close_vector_store()

    if report.applied_min_score is None:
        threshold_analysis = calibrate_score_threshold(report.case_results).as_dict()
    else:
        threshold_metrics = calculate_score_threshold_metrics(
            report.case_results,
            report.applied_min_score,
        )
        threshold_analysis = {
            "mode": "validation",
            "status": "evaluated",
            "applied_min_score": report.applied_min_score,
            "metrics": threshold_metrics.as_dict(),
        }

    output = {
        "mode": "evaluate",
        "schema_version": dataset.schema_version,
        "dataset": str(args.cases),
        "description": dataset.description,
        "vector_store_backend": backend_name,
        "embedding_model": get_embedding_model(),
        **report.as_dict(),
        "score_threshold_analysis": threshold_analysis,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


def _positive_integer(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def _cosine_score(raw_value: str) -> float:
    try:
        return validate_min_score(raw_value, setting_name="--min-score")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Gro AI catalog retrieval against labeled cases"
    )
    parser.add_argument(
        "--cases",
        type=Path,
        required=True,
        help="path to a versioned retrieval evaluation JSON file",
    )
    parser.add_argument(
        "--top-k",
        type=_positive_integer,
        default=5,
        help="number of retrieved items scored per case (default: 5)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate cases without embedding queries or searching a store",
    )
    parser.add_argument(
        "--min-score",
        type=_cosine_score,
        default=None,
        help=(
            "apply and evaluate a fixed cosine threshold from -1 to 1; "
            "omit to calibrate a recommendation"
        ),
    )
    return parser


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = _build_parser()
    try:
        asyncio.run(_run_cli(parser.parse_args()))
    except Exception as exc:
        parser.exit(1, f"Retrieval evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
