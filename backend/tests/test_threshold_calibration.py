import unittest
from types import SimpleNamespace

from vector.threshold_calibration import (
    calculate_score_threshold_metrics,
    calibrate_score_threshold,
)


def labeled_result(
    relevant_item_ids: tuple[int, ...],
    hits: list[tuple[float, bool]],
) -> SimpleNamespace:
    return SimpleNamespace(
        relevant_item_ids=relevant_item_ids,
        hits=[
            SimpleNamespace(score=score, relevant=relevant) for score, relevant in hits
        ],
    )


class ScoreThresholdMetricsTests(unittest.TestCase):
    def test_counts_unretrieved_labels_as_false_negatives(self):
        result = labeled_result(
            (10, 20, 30),
            [(0.90, True), (0.85, False), (0.80, True), (0.70, False)],
        )

        metrics = calculate_score_threshold_metrics([result], 0.80)

        self.assertEqual(metrics.returned_count, 3)
        self.assertEqual(metrics.true_positive_count, 2)
        self.assertEqual(metrics.false_positive_count, 1)
        self.assertEqual(metrics.false_negative_count, 1)
        self.assertAlmostEqual(metrics.precision, 2 / 3)
        self.assertAlmostEqual(metrics.recall, 2 / 3)
        self.assertAlmostEqual(metrics.f1, 2 / 3)


class ScoreThresholdCalibrationTests(unittest.TestCase):
    def test_recommends_observed_threshold_with_highest_global_f1(self):
        result = labeled_result(
            (10, 20, 30),
            [(0.90, True), (0.85, False), (0.80, True), (0.70, False)],
        )

        calibration = calibrate_score_threshold([result])

        self.assertEqual(calibration.status, "recommended")
        self.assertEqual(calibration.candidate_threshold_count, 4)
        self.assertIsNotNone(calibration.recommendation)
        assert calibration.recommendation is not None
        self.assertEqual(calibration.recommendation.min_score, 0.80)
        self.assertAlmostEqual(calibration.recommendation.f1, 2 / 3)

    def test_equal_f1_prefers_higher_recall(self):
        result = labeled_result(
            (10, 20),
            [(0.90, True), (0.80, True), (0.80, False), (0.80, False)],
        )

        calibration = calibrate_score_threshold([result])

        assert calibration.recommendation is not None
        self.assertEqual(calibration.recommendation.min_score, 0.80)
        self.assertEqual(calibration.recommendation.recall, 1.0)

    def test_requires_both_relevant_and_non_relevant_retrieved_candidates(self):
        only_relevant = calibrate_score_threshold(
            [labeled_result((10,), [(0.90, True)])]
        )
        no_relevant = calibrate_score_threshold(
            [labeled_result((10,), [(0.90, False)])]
        )

        self.assertEqual(only_relevant.status, "unavailable")
        self.assertIn("No non-relevant", only_relevant.reason or "")
        self.assertEqual(no_relevant.status, "unavailable")
        self.assertIn("No labeled-relevant", no_relevant.reason or "")


if __name__ == "__main__":
    unittest.main()
