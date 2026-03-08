import unittest
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_synthesis.smoothness_weighted_selection import (
    SelectionWeights,
    compute_motion_smoothness,
    format_ranked_results_table,
    rank_candidate_results,
    resolve_selection_weights,
)


def make_fake_result(label, dtw_distance, coupler_trajectory, metadata=None, constrained_length=None, constrained_angle=None):
    return SimpleNamespace(
        dtw_distance=dtw_distance,
        coupler_trajectory=np.asarray(coupler_trajectory, dtype=np.float64),
        candidate=SimpleNamespace(label=label, metadata={'mode': 'test'} if metadata is None else metadata),
        mechanism_type='RRRR',
        bar_type='4bar',
        coords_string='_0.000_0.000_1.000_0.000_1.000_1.000_RRRR',
        mechanism_params=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        constrained_length=constrained_length,
        constrained_angle=constrained_angle,
    )


class SmoothnessMetricTests(unittest.TestCase):
    def test_smooth_line_is_smoother_than_zig_zag(self):
        smooth_line = np.array(
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0]],
            dtype=np.float64,
        )
        zig_zag = np.array(
            [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 1.0], [0.0, 1.5]],
            dtype=np.float64,
        )

        smooth_metrics = compute_motion_smoothness(smooth_line)
        zig_zag_metrics = compute_motion_smoothness(zig_zag)

        self.assertLess(smooth_metrics.smoothness_loss, zig_zag_metrics.smoothness_loss)
        self.assertLess(smooth_metrics.jerk_rms, zig_zag_metrics.jerk_rms)


class WeightResolutionTests(unittest.TestCase):
    def test_preset_smoothness_weights_bias_smoothness(self):
        weights = SelectionWeights.from_mode('smoothness').normalized()
        self.assertGreater(weights.smoothness, weights.curve_following)

    def test_custom_weights_are_normalized(self):
        weights = resolve_selection_weights(curve_weight=2.0, smoothness_weight=1.0)
        self.assertAlmostEqual(weights.curve_following, 2.0 / 3.0)
        self.assertAlmostEqual(weights.smoothness, 1.0 / 3.0)


class RankingTests(unittest.TestCase):
    def test_curve_focused_ranking_prefers_better_dtw(self):
        smooth_candidate = make_fake_result(
            'smooth',
            0.45,
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0]],
        )
        curve_candidate = make_fake_result(
            'curve',
            0.10,
            [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 1.0], [0.0, 1.5]],
        )

        ranked = rank_candidate_results(
            [smooth_candidate, curve_candidate],
            SelectionWeights.from_mode('curve'),
        )
        self.assertEqual(ranked[0].result.candidate.label, 'curve')

    def test_smoothness_focused_ranking_prefers_smoother_motion(self):
        smooth_candidate = make_fake_result(
            'smooth',
            0.45,
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0]],
        )
        curve_candidate = make_fake_result(
            'curve',
            0.10,
            [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 1.0], [0.0, 1.5]],
        )

        ranked = rank_candidate_results(
            [smooth_candidate, curve_candidate],
            SelectionWeights.from_mode('smoothness'),
        )
        self.assertEqual(ranked[0].result.candidate.label, 'smooth')

    def test_crank_table_includes_requested_and_generated_metrics(self):
        crank_candidate = make_fake_result(
            'crank',
            0.20,
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0]],
            metadata={
                'mode': 'crank-length',
                'length': 0.5,
                'angle': 90.0,
                'sampled_point': (0.0, 0.5),
                'mechanism_type_prefix': 'RRRR',
            },
            constrained_length=0.55,
            constrained_angle=84.0,
        )
        ranked = rank_candidate_results([crank_candidate], SelectionWeights.from_mode('curve'))
        table = format_ranked_results_table(ranked, top_rows=1)

        self.assertIn('ReqLen', table)
        self.assertIn('GenLen', table)
        self.assertIn('0.500', table)
        self.assertIn('0.550', table)


if __name__ == '__main__':
    unittest.main()
