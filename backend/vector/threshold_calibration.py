"""Calibrate a global vector-search threshold from labeled retrieval hits."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from vector.search_policy import validate_min_score


class LabeledHit(Protocol):
    score: float
    relevant: bool


class LabeledCaseResult(Protocol):
    relevant_item_ids: Sequence[int]
    hits: Sequence[LabeledHit]


@dataclass(frozen=True, slots=True)
class ScoreThresholdMetrics:
    """Binary relevance counts and metrics at one score threshold."""

    min_score: float | None
    returned_count: int
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, object]:
        return {
            "min_score": self.min_score,
            "returned_count": self.returned_count,
            "true_positive_count": self.true_positive_count,
            "false_positive_count": self.false_positive_count,
            "false_negative_count": self.false_negative_count,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True, slots=True)
class ScoreThresholdCalibration:
    """A recommendation, or an explanation of why one is unavailable."""

    status: str
    candidate_threshold_count: int
    total_relevant_label_count: int
    retrieved_relevant_candidate_count: int
    retrieved_non_relevant_candidate_count: int
    recommendation: ScoreThresholdMetrics | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": "calibration",
            "status": self.status,
            "selection_objective": "maximum_f1",
            "tie_break_order": [
                "higher_recall",
                "higher_precision",
                "lower_min_score",
            ],
            "candidate_threshold_count": self.candidate_threshold_count,
            "total_relevant_label_count": self.total_relevant_label_count,
            "retrieved_relevant_candidate_count": (
                self.retrieved_relevant_candidate_count
            ),
            "retrieved_non_relevant_candidate_count": (
                self.retrieved_non_relevant_candidate_count
            ),
            "recommended_min_score": (
                self.recommendation.min_score if self.recommendation else None
            ),
            "metrics_at_recommended_threshold": (
                self.recommendation.as_dict() if self.recommendation else None
            ),
            "reason": self.reason,
        }


def calculate_score_threshold_metrics(
    case_results: Sequence[LabeledCaseResult],
    min_score: float | None,
) -> ScoreThresholdMetrics:
    """Treat retained hits as predictions and all labeled IDs as positives."""

    threshold = validate_min_score(min_score) if min_score is not None else None
    total_relevant = sum(len(result.relevant_item_ids) for result in case_results)
    if total_relevant <= 0:
        raise ValueError("Threshold metrics require at least one relevant label")

    true_positives = 0
    false_positives = 0
    for result in case_results:
        for hit in result.hits:
            score = float(hit.score)
            if not math.isfinite(score):
                raise ValueError("Threshold calibration requires finite hit scores")
            if threshold is not None and score < threshold:
                continue
            if hit.relevant:
                true_positives += 1
            else:
                false_positives += 1

    returned_count = true_positives + false_positives
    false_negatives = total_relevant - true_positives
    precision = true_positives / returned_count if returned_count else 0.0
    recall = true_positives / total_relevant
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ScoreThresholdMetrics(
        min_score=threshold,
        returned_count=returned_count,
        true_positive_count=true_positives,
        false_positive_count=false_positives,
        false_negative_count=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def calibrate_score_threshold(
    case_results: Sequence[LabeledCaseResult],
) -> ScoreThresholdCalibration:
    """Choose the observed cosine threshold with the strongest global F1."""

    total_relevant = sum(len(result.relevant_item_ids) for result in case_results)
    relevant_scores: list[float] = []
    non_relevant_scores: list[float] = []
    for result in case_results:
        for hit in result.hits:
            score = validate_min_score(
                hit.score,
                setting_name="retrieval evaluation hit score",
            )
            if hit.relevant:
                relevant_scores.append(score)
            else:
                non_relevant_scores.append(score)

    candidate_thresholds = sorted(set(relevant_scores + non_relevant_scores))
    common_fields = {
        "candidate_threshold_count": len(candidate_thresholds),
        "total_relevant_label_count": total_relevant,
        "retrieved_relevant_candidate_count": len(relevant_scores),
        "retrieved_non_relevant_candidate_count": len(non_relevant_scores),
    }
    if not relevant_scores:
        return ScoreThresholdCalibration(
            status="unavailable",
            reason=(
                "No labeled-relevant item was retrieved; improve retrieval or "
                "increase top_k before calibrating a threshold"
            ),
            **common_fields,
        )
    if not non_relevant_scores:
        return ScoreThresholdCalibration(
            status="unavailable",
            reason=(
                "No non-relevant candidate was retrieved; both positive and "
                "negative examples are required to calibrate a threshold"
            ),
            **common_fields,
        )

    candidates = [
        calculate_score_threshold_metrics(case_results, threshold)
        for threshold in candidate_thresholds
    ]
    recommendation = max(
        candidates,
        key=lambda metrics: (
            metrics.f1,
            metrics.recall,
            metrics.precision,
            -float(metrics.min_score),
        ),
    )
    return ScoreThresholdCalibration(
        status="recommended",
        recommendation=recommendation,
        **common_fields,
    )
