"""Shared score policy for catalog vector retrieval."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

from vector.store import SearchHit


VECTOR_SEARCH_MIN_SCORE_ENV = "VECTOR_SEARCH_MIN_SCORE"


def validate_min_score(
    value: object,
    *,
    setting_name: str = "minimum similarity score",
) -> float:
    """Return a valid cosine-similarity threshold."""

    if isinstance(value, bool):
        raise ValueError(f"{setting_name} must be a number between -1 and 1")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{setting_name} must be a number between -1 and 1") from exc
    if not math.isfinite(score) or not -1.0 <= score <= 1.0:
        raise ValueError(f"{setting_name} must be a finite number between -1 and 1")
    return score


def parse_optional_min_score(
    raw_value: str | None,
    *,
    setting_name: str = "minimum similarity score",
) -> float | None:
    """Parse an optional score threshold; blank values disable filtering."""

    if raw_value is None or not raw_value.strip():
        return None
    return validate_min_score(raw_value.strip(), setting_name=setting_name)


def get_vector_search_min_score() -> float | None:
    """Read the optional production search threshold from the environment."""

    return parse_optional_min_score(
        os.getenv(VECTOR_SEARCH_MIN_SCORE_ENV),
        setting_name=VECTOR_SEARCH_MIN_SCORE_ENV,
    )


def filter_search_hits(
    hits: Sequence[SearchHit],
    min_score: float | None,
) -> list[SearchHit]:
    """Keep hits whose cosine score meets an inclusive minimum threshold."""

    if min_score is None:
        return list(hits)
    threshold = validate_min_score(min_score)
    return [hit for hit in hits if float(hit.score) >= threshold]
