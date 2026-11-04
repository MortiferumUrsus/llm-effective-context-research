"""Checks for misleading capacity summaries and invalid paired comparisons."""
import unittest

import pandas as pd

from analysis.final_report import collapse_source_conditions, grid_pattern, load_observations
from analysis.research_report import analysis_population, matched, wilson


class EffectiveContextChecks(unittest.TestCase):
    def test_last_success_does_not_hide_intermediate_failure(self):
        self.assertEqual(grid_pattern([.53, .30, .27, .40, .60]),
                         "later_recovery_or_nonmonotone_threshold_pattern")
        self.assertEqual(grid_pattern([1, 1, 1]), "all_observed_means_at_or_above_threshold")
        self.assertEqual(grid_pattern([.2, .3, .4]), "all_observed_means_below_threshold")

    def test_perfect_small_sample_is_not_certain(self):
        low, high = wilson(20, 20)
        self.assertAlmostEqual(float(low), .8388748419471806)
        self.assertAlmostEqual(float(high), 1.)

    def test_repeated_input_scores_are_averaged_without_outcome_selection(self):
        mapping = {"same": {"config": "32k", "row_index": 4, "prefix_family_id": 4}}
        base = {"dataset_t": "BABILong_qa2", "model": "test", "procedure_A": "A0_baseline",
                "source_task_id": "4", "cluster_id": "babi-prefix-4", "input_text_hash": "same",
                "gold_answer": "garden", "actual_words": 22000}
        frame = pd.DataFrame([base | {"metric_value": 0}, base | {"metric_value": 1}])
        result = collapse_source_conditions(frame, mapping)
        self.assertEqual(result.score.tolist(), [.5])
        self.assertEqual(result.repeats.tolist(), [2])
        frame.loc[1, "gold_answer"] = "kitchen"
        with self.assertRaises(ValueError):
            collapse_source_conditions(frame, mapping)

    def test_answer_failure_is_not_removed_with_provider_failures(self):
        base = {"dataset_t": "NIAH", "model": "test", "target_words": 6000,
                "seed": 42, "procedure_A": "A0_baseline", "metric_value": 0,
                "provider_fail": False, "finish_reason": "parse_error"}
        frame = pd.DataFrame([base | {"episode_idx": 0},
                              base | {"episode_idx": 1, "provider_fail": True}])
        attempted, retained = analysis_population(frame)
        self.assertEqual(len(attempted), 2)
        self.assertEqual(retained.episode_idx.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
