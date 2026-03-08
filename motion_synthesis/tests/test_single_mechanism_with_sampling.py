import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference_mechanism import MechanismInference
from motion_synthesis.single_mechanism_with_sampling import (
    CrankLengthSamplingStrategy,
    PointPrefix,
    build_circle_angles,
    dtw_path,
    fixed_point_indices_for_optimization,
    generated_point_index_to_full_index,
)


class CircleSamplingTests(unittest.TestCase):
    def test_build_circle_angles_respects_step(self):
        self.assertEqual(build_circle_angles(90.0), [0.0, 90.0, 180.0, 270.0])

    def test_crank_length_strategy_expands_lengths_and_angles(self):
        strategy = CrankLengthSamplingStrategy(
            lengths=[0.5, 1.0],
            crank_angles=[0.0, 90.0],
            mechanism_type='RRRR',
        )

        candidates = list(strategy.build_candidates({'mechanism_type': 'Steph1T1'}))

        self.assertEqual(len(candidates), 4)
        self.assertEqual(candidates[0].metadata['mode'], 'crank-length')
        self.assertEqual(candidates[0].mechanism_type_prefix, 'RRRR')
        self.assertAlmostEqual(candidates[0].point_prefixes[0].x, 0.5, places=5)
        self.assertAlmostEqual(candidates[0].point_prefixes[0].y, 0.0, places=5)
        self.assertAlmostEqual(candidates[1].point_prefixes[0].x, 0.0, places=5)
        self.assertAlmostEqual(candidates[1].point_prefixes[0].y, 0.5, places=5)

    def test_crank_length_strategy_uses_sample_mechanism_type_by_default(self):
        strategy = CrankLengthSamplingStrategy(lengths=[0.5], crank_angles=[0.0])
        candidate = next(iter(strategy.build_candidates({'mechanism_type': 'Watt2T1A1'})))
        self.assertEqual(candidate.mechanism_type_prefix, 'Watt2T1A1')


class TopologyMappingTests(unittest.TestCase):
    def test_generated_point_index_maps_back_for_rrrr(self):
        self.assertEqual(generated_point_index_to_full_index('RRRR', 0), 1)
        self.assertEqual(generated_point_index_to_full_index('RRRR', 1), 3)
        self.assertEqual(generated_point_index_to_full_index('RRRR', 2), 4)

    def test_fixed_points_include_ground_and_prefixed_crank_point(self):
        fixed_points = fixed_point_indices_for_optimization(
            'RRRR',
            (PointPrefix(index=0, x=0.5, y=0.0),),
        )
        self.assertEqual(fixed_points, [0, 1, 2])


class PrefixTokenTests(unittest.TestCase):
    def setUp(self):
        self.inference = MechanismInference.__new__(MechanismInference)
        self.inference.vocab = {
            'MECH_TYPE:': 1,
            'RRRR': 2,
            'POINTS:': 3,
            'P0': 4,
            'X:': 5,
            'Y:': 6,
            'BIN_100': 7,
            'BIN_105': 8,
        }

    def test_build_prefix_tokens_serializes_mechanism_and_point(self):
        tokens = self.inference.build_prefix_tokens(
            mech_type='RRRR',
            point_prefixes=[(0, 0.0, 0.5)],
        )
        self.assertEqual(tokens, [1, 2, 3, 4, 5, 7, 6, 8])

    def test_build_prefix_tokens_requires_mechanism_type(self):
        with self.assertRaises(ValueError):
            self.inference.build_prefix_tokens(point_prefixes=[(0, 0.0, 0.5)])


class DtwTests(unittest.TestCase):
    def test_dtw_distance_is_zero_for_identical_curves(self):
        curve = [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0)]
        path, distance = dtw_path(curve, curve)
        self.assertGreaterEqual(len(path), len(curve))
        self.assertAlmostEqual(distance, 0.0, places=7)


if __name__ == '__main__':
    unittest.main()
