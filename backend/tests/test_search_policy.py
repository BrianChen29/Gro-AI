import os
import unittest
from unittest.mock import patch

from vector.search_policy import (
    filter_search_hits,
    get_vector_search_min_score,
    parse_optional_min_score,
    validate_min_score,
)
from vector.store import SearchHit


class SearchPolicyTests(unittest.TestCase):
    def test_unset_or_blank_environment_value_disables_filtering(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(get_vector_search_min_score())
        with patch.dict(os.environ, {"VECTOR_SEARCH_MIN_SCORE": "  "}, clear=True):
            self.assertIsNone(get_vector_search_min_score())

    def test_parses_valid_cosine_threshold(self):
        self.assertEqual(parse_optional_min_score(" 0.72 "), 0.72)
        self.assertEqual(validate_min_score(-1), -1.0)
        self.assertEqual(validate_min_score(1), 1.0)

    def test_rejects_non_numeric_non_finite_and_out_of_range_thresholds(self):
        for raw_value in ("not-a-number", "nan", "inf", "1.01", "-1.01"):
            with self.subTest(raw_value=raw_value):
                with self.assertRaisesRegex(ValueError, "between -1 and 1"):
                    parse_optional_min_score(raw_value)
        with self.assertRaises(ValueError):
            validate_min_score(True)

    def test_filters_inclusively_without_reordering_hits(self):
        hits = [
            SearchHit(item_id=10, score=0.91),
            SearchHit(item_id=20, score=0.80),
            SearchHit(item_id=30, score=0.79),
        ]

        self.assertEqual(
            [hit.item_id for hit in filter_search_hits(hits, 0.80)],
            [10, 20],
        )
        self.assertEqual(filter_search_hits(hits, None), hits)


if __name__ == "__main__":
    unittest.main()
