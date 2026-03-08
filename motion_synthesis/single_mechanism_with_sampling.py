from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize

try:
    from tslearn.metrics import dtw_path
except ImportError:
    def dtw_path(curve1, curve2):
        """Fallback DTW implementation when tslearn is unavailable."""
        curve1 = np.asarray(curve1, dtype=np.float64)
        curve2 = np.asarray(curve2, dtype=np.float64)

        len_curve1 = len(curve1)
        len_curve2 = len(curve2)
        cumulative_cost = np.full((len_curve1 + 1, len_curve2 + 1), np.inf, dtype=np.float64)
        predecessors = np.zeros((len_curve1, len_curve2, 2), dtype=np.int32)
        cumulative_cost[0, 0] = 0.0

        for i in range(1, len_curve1 + 1):
            for j in range(1, len_curve2 + 1):
                point_cost = np.linalg.norm(curve1[i - 1] - curve2[j - 1])
                candidate_costs = (
                    cumulative_cost[i - 1, j],
                    cumulative_cost[i, j - 1],
                    cumulative_cost[i - 1, j - 1],
                )
                best_candidate = int(np.argmin(candidate_costs))
                if best_candidate == 0:
                    predecessor = (i - 1, j)
                elif best_candidate == 1:
                    predecessor = (i, j - 1)
                else:
                    predecessor = (i - 1, j - 1)

                cumulative_cost[i, j] = point_cost + candidate_costs[best_candidate]
                predecessors[i - 1, j - 1] = predecessor

        i, j = len_curve1, len_curve2
        path = []
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            i, j = predecessors[i - 1, j - 1]
        while i > 0:
            i -= 1
            path.append((i, 0))
        while j > 0:
            j -= 1
            path.append((0, j))
        path.reverse()

        return path, float(cumulative_cost[len_curve1, len_curve2])

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from ground_joint_utils import GroundJointNormalizer
from inference_mechanism import MechanismInference
from wrapper.mechanism_core import MechanismAnimator
from wrapper.mechanism_wrapper import MechanismWrapper

FOUR_BAR_TYPES = {'RRRR', 'PRPR', 'RRPR', 'RRRP', 'RPPR', 'RRPP'}
DEFAULT_AFFINE_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
DEFAULT_AFFINE_TRANSLATIONS = [
    (0.0, 0.0),
    (0.3, 0.0),
    (-0.3, 0.0),
    (0.0, 0.3),
    (0.0, -0.3),
    (0.3, 0.3),
    (-0.3, -0.3),
]
DEFAULT_CRANK_LENGTHS = [0.5, 0.75, 1.0]
DEFAULT_CRANK_ANGLE_STEP = 30.0
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'output'
GROUND_JOINT_NORMALIZER = GroundJointNormalizer(str(PROJECT_ROOT / 'wrapper' / 'BSIdict_468.json'))


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass(frozen=True)
class PointPrefix:
    """A prefixed generated point in normalized mechanism coordinates."""

    index: int
    x: float
    y: float

    def as_tuple(self):
        return self.index, self.x, self.y


@dataclass(frozen=True)
class AffineTransformSpec:
    """Affine transform applied to the target curve/control points before generation."""

    angle_degrees: float = 0.0
    translation: tuple[float, float] = (0.0, 0.0)

    def apply(self, points):
        tx, ty = self.translation
        return apply_affine_transform(points, self.angle_degrees, tx, ty)

    def invert(self, points):
        tx, ty = self.translation
        return inverse_affine_transform(points, self.angle_degrees, tx, ty)

    def summary(self):
        tx, ty = self.translation
        return f"angle={self.angle_degrees:.1f}°, t=({tx:.2f},{ty:.2f})"


@dataclass
class SamplingCandidate:
    """Single sampling hypothesis evaluated by the search loop."""

    label: str
    transform: AffineTransformSpec = field(default_factory=AffineTransformSpec)
    mechanism_type_prefix: Optional[str] = None
    point_prefixes: tuple[PointPrefix, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def transform_control_points(self, control_points):
        return self.transform.apply(control_points)

    def transform_target_curve(self, target_curve):
        return self.transform.apply(target_curve)

    def restore_coupler_trajectory(self, coupler_traj):
        return self.transform.invert(coupler_traj)


@dataclass
class SimulationOutcome:
    coupler_trajectory: np.ndarray
    wrapper: MechanismWrapper
    bar_type: str
    coords_string: str


@dataclass
class SearchResult:
    candidate: SamplingCandidate
    target_curve: np.ndarray
    transformed_target_curve: np.ndarray
    mechanism_type: str
    mechanism_params: list[list[float]]
    coupler_trajectory: np.ndarray
    transformed_coupler_trajectory: np.ndarray
    wrapper: MechanismWrapper
    bar_type: str
    coords_string: str
    dtw_distance: float
    constrained_full_point_index: Optional[int] = None
    constrained_point: Optional[np.ndarray] = None
    constrained_length: Optional[float] = None
    constrained_angle: Optional[float] = None
    optimized_coords: Optional[str] = None
    optimized_dtw: Optional[float] = None
    pre_optimization_dtw: Optional[float] = None
    optimized_coupler_trajectory: Optional[np.ndarray] = None
    optimized_dtw_original: Optional[float] = None
    wrapper_optimized: Optional[MechanismWrapper] = None


# ============================================================================
# SAMPLING STRATEGIES
# ============================================================================

class SamplingStrategy:
    name = 'sampling'

    def describe(self):
        return self.name

    def build_candidates(self, sample) -> Iterable[SamplingCandidate]:
        raise NotImplementedError


@dataclass
class AffineSamplingStrategy(SamplingStrategy):
    angles: Sequence[float] = field(default_factory=lambda: list(DEFAULT_AFFINE_ANGLES))
    translations: Sequence[tuple[float, float]] = field(
        default_factory=lambda: list(DEFAULT_AFFINE_TRANSLATIONS)
    )
    name: str = 'affine'

    def describe(self):
        return f"{len(self.angles)} rotations x {len(self.translations)} translations"

    def build_candidates(self, sample):
        for angle in self.angles:
            for translation in self.translations:
                yield SamplingCandidate(
                    label=f"angle={float(angle):.1f}°, t=({float(translation[0]):.2f},{float(translation[1]):.2f})",
                    transform=AffineTransformSpec(float(angle), (float(translation[0]), float(translation[1]))),
                    metadata={
                        'mode': 'affine',
                        'angle': float(angle),
                        'translation': (float(translation[0]), float(translation[1])),
                    },
                )


@dataclass
class CrankLengthSamplingStrategy(SamplingStrategy):
    lengths: Sequence[float] = field(default_factory=lambda: list(DEFAULT_CRANK_LENGTHS))
    crank_angles: Sequence[float] = field(
        default_factory=lambda: build_circle_angles(DEFAULT_CRANK_ANGLE_STEP)
    )
    mechanism_type: Optional[str] = None
    name: str = 'crank-length'

    def describe(self):
        return f"{len(self.lengths)} lengths x {len(self.crank_angles)} circle samples"

    def _resolve_mechanism_type(self, sample):
        mech_type = self.mechanism_type or sample.get('mechanism_type')
        if not mech_type:
            raise ValueError(
                'Crank-length sampling requires a mechanism type prefix. '
                'Pass mechanism_type or provide a sample with mechanism_type.'
            )
        return mech_type

    def build_candidates(self, sample):
        mech_type = self._resolve_mechanism_type(sample)
        for length in self.lengths:
            if length < 0:
                raise ValueError(f'Crank length must be non-negative, got {length}.')
            for angle in self.crank_angles:
                x_coord, y_coord = point_on_circle(float(length), float(angle))
                yield SamplingCandidate(
                    label=f"length={float(length):.3f}, theta={float(angle):.1f}°",
                    mechanism_type_prefix=mech_type,
                    point_prefixes=(PointPrefix(index=0, x=x_coord, y=y_coord),),
                    metadata={
                        'mode': 'crank-length',
                        'length': float(length),
                        'angle': float(angle),
                        'sampled_point': (x_coord, y_coord),
                        'mechanism_type_prefix': mech_type,
                    },
                )


# ============================================================================
# AFFINE / GEOMETRY UTILITIES
# ============================================================================

def rotate_points(points, angle_degrees):
    """Rotate points about the origin by a given angle in degrees."""
    angle = np.radians(angle_degrees)
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float32,
    )
    return np.asarray(points, dtype=np.float32) @ rotation_matrix



def translate_points(points, tx, ty):
    """Translate points by (tx, ty)."""
    return np.asarray(points, dtype=np.float32) + np.array([tx, ty], dtype=np.float32)



def apply_affine_transform(points, angle_degrees, tx, ty):
    """Apply rotation then translation to points."""
    points = np.asarray(points, dtype=np.float32)
    rotated = rotate_points(points, angle_degrees)
    return translate_points(rotated, tx, ty)



def inverse_affine_transform(points, angle_degrees, tx, ty):
    """Undo translation then rotation."""
    points = np.asarray(points, dtype=np.float32)
    untranslated = translate_points(points, -tx, -ty)
    return rotate_points(untranslated, -angle_degrees)



def point_on_circle(length, angle_degrees, center=(0.0, 0.0)):
    """Sample a point on a circle of the requested radius in normalized space."""
    center = np.asarray(center, dtype=np.float32)
    angle_radians = np.radians(angle_degrees)
    point = np.array(
        [length * np.cos(angle_radians), length * np.sin(angle_radians)],
        dtype=np.float32,
    )
    return tuple((center + point).tolist())



def build_circle_angles(angle_step):
    """Build a full set of circle samples using a fixed angular step."""
    if angle_step <= 0:
        raise ValueError(f'angle_step must be positive, got {angle_step}.')
    return [float(angle) for angle in np.arange(0.0, 360.0, angle_step, dtype=np.float32)]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_processed_data(pickle_path=None):
    """Load processed training/validation data."""
    if pickle_path is None:
        pickle_path = PROJECT_ROOT / 'processed_data.pkl'
    else:
        pickle_path = Path(pickle_path)

    print(f"Loading data from: {pickle_path}")
    with open(pickle_path, 'rb') as handle:
        data = pickle.load(handle)
    print(f"  Loaded {len(data['processed_data'])} samples")
    return data



def create_validation_split(dataset, val_split=0.1, seed=42):
    """Create a stratified validation split."""
    np.random.seed(seed)

    mechanism_groups = {}
    for idx, sample in enumerate(dataset['processed_data']):
        mechanism_groups.setdefault(sample['mechanism_type'], []).append(idx)

    train_indices, val_indices = [], []
    for indices in mechanism_groups.values():
        np.random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_split))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    np.random.shuffle(val_indices)
    return np.array(train_indices), np.array(val_indices)



def get_sample(data, val_indices, val_index):
    """Get a validation sample by index."""
    actual_index = val_indices[val_index]
    return data['processed_data'][actual_index]


# ============================================================================
# CURVE UTILITIES
# ============================================================================

def reconstruct_curve_from_control_points(control_points, num_points=200):
    """Reconstruct a curve from B-spline control points."""
    control_points = np.asarray(control_points, dtype=float)
    degree = 3
    num_control_points = len(control_points)

    if num_control_points <= degree:
        return None

    knot_vector = np.concatenate(
        (
            np.zeros(degree),
            np.linspace(0, 1, num_control_points - degree + 1),
            np.ones(degree),
        )
    )

    bspline_x = BSpline(knot_vector, control_points[:, 0], degree)
    bspline_y = BSpline(knot_vector, control_points[:, 1], degree)
    t_vals = np.linspace(0, 1, num_points)
    return np.vstack([bspline_x(t_vals), bspline_y(t_vals)]).T



def compute_dtw_distance(curve1, curve2):
    """Compute DTW distance between two curves."""
    try:
        _, distance = dtw_path(curve1, curve2)
        if np.isnan(distance) or np.isinf(distance):
            return float('inf')
        return float(distance)
    except Exception:
        return float('inf')



def get_normalization_params(curve):
    """Get normalization parameters from a curve."""
    curve_array = np.asarray(curve).squeeze()
    mean = np.mean(curve_array, axis=0)
    centered = curve_array - mean
    rms_var = np.sqrt(np.var(centered[:, 0]) + np.var(centered[:, 1]))
    return mean, rms_var if rms_var != 0 else 1.0



def apply_normalization(curve, mean, rms_var):
    """Apply a fixed normalization to a curve."""
    curve_array = np.asarray(curve).squeeze()
    return (curve_array - mean) / rms_var



def normalize_curves(coupler_traj, target_traj):
    """Normalize both curves using the target trajectory statistics."""
    mean, rms_var = get_normalization_params(target_traj)
    normalized_target = apply_normalization(target_traj, mean, rms_var)
    normalized_coupler = apply_normalization(coupler_traj, mean, rms_var)
    return normalized_coupler, normalized_target, mean, rms_var


# ============================================================================
# TOPOLOGY / PREFIX HELPERS
# ============================================================================

def get_bar_type(mechanism_type):
    """Infer high-level mechanism family for reporting/simulation setup."""
    if mechanism_type in FOUR_BAR_TYPES:
        return '4bar'
    if mechanism_type.startswith('Steph') or mechanism_type.startswith('Watt'):
        return '6bar'
    if mechanism_type.startswith('Type'):
        return '8bar'
    return None



def get_removed_ground_indices(mechanism_type):
    ground_indices = GROUND_JOINT_NORMALIZER.get_ground_joint_indices(mechanism_type)
    return ground_indices[:2] if len(ground_indices) >= 3 else ground_indices



def get_kept_point_indices(mechanism_type):
    total_points = len(GROUND_JOINT_NORMALIZER.bsi_dict[mechanism_type]['B'][0])
    removed_indices = set(get_removed_ground_indices(mechanism_type))
    return [index for index in range(total_points) if index not in removed_indices]



def generated_point_index_to_full_index(mechanism_type, generated_point_index):
    kept_indices = get_kept_point_indices(mechanism_type)
    if generated_point_index < 0 or generated_point_index >= len(kept_indices):
        raise IndexError(
            f'Generated point index {generated_point_index} is out of range for {mechanism_type}.'
        )
    return kept_indices[generated_point_index]



def fixed_point_indices_for_optimization(mechanism_type, point_prefixes=None):
    fixed_points = set(GROUND_JOINT_NORMALIZER.get_ground_joint_indices(mechanism_type))
    if point_prefixes:
        fixed_points.update(
            generated_point_index_to_full_index(mechanism_type, prefix.index)
            for prefix in point_prefixes
        )
    return sorted(fixed_points)



def extract_constrained_point_metrics(mechanism_type, mechanism_params, point_prefixes):
    if not point_prefixes:
        return None, None, None, None

    full_index = generated_point_index_to_full_index(mechanism_type, point_prefixes[0].index)
    if full_index >= len(mechanism_params):
        return full_index, None, None, None

    point = np.asarray(mechanism_params[full_index], dtype=np.float32)
    length = float(np.linalg.norm(point))
    angle = float(np.degrees(np.arctan2(point[1], point[0])))
    return full_index, point, length, angle


# ============================================================================
# MODEL GENERATION / SIMULATION
# ============================================================================

def mechanism_points_for_simulation(mechanism_params, mechanism_type):
    bar_type = get_bar_type(mechanism_type)
    if bar_type == '4bar':
        return mechanism_params[:5], bar_type
    if bar_type in {'6bar', '8bar'}:
        return mechanism_params, bar_type
    return None, None



def build_coords_string(mechanism_params, mechanism_type):
    coords_to_use, _ = mechanism_points_for_simulation(mechanism_params, mechanism_type)
    if coords_to_use is None:
        return None
    flat_coords = [f"{float(value):.3f}" for pair in coords_to_use for value in pair]
    return '_' + '_'.join(flat_coords) + f'_{mechanism_type}'



def generate_mechanism_from_curve(
    inference,
    control_points,
    temperature=0.001,
    top_k=5,
    mech_type=None,
    point_prefixes=None,
):
    """Generate a mechanism from control points with optional autoregressive prefixes."""
    prefix_payload = None
    if point_prefixes:
        prefix_payload = [prefix.as_tuple() for prefix in point_prefixes]

    return inference.generate_mechanism_params(
        control_points,
        temperature=temperature,
        mech_type=mech_type,
        top_k=top_k,
        process_curve=False,
        point_prefixes=prefix_payload,
    )



def simulate_mechanism(mechanism_params, mechanism_type):
    """Simulate a generated mechanism and return its coupler trajectory."""
    coords_string = build_coords_string(mechanism_params, mechanism_type)
    coords_to_use, bar_type = mechanism_points_for_simulation(mechanism_params, mechanism_type)
    if coords_string is None or coords_to_use is None:
        return None

    wrapper = MechanismWrapper(coords_string)
    wrapper.simulate(
        speed_scale=1.0,
        steps=200,
        relative_tolerance=0.1,
        driving_element=1,
        start_angle=0,
        end_angle=360,
    )

    if len(wrapper.poses) <= 1:
        return None

    coupler_traj = wrapper.get_coupler_trajectory()
    if not coupler_traj:
        return None

    return SimulationOutcome(
        coupler_trajectory=np.asarray(coupler_traj, dtype=np.float32),
        wrapper=wrapper,
        bar_type=bar_type,
        coords_string=coords_string,
    )



def evaluate_sampling_candidate(inference, control_points, target_curve, candidate, temperature):
    """Run generation, simulation, and DTW scoring for a single candidate."""
    transformed_control_points = candidate.transform_control_points(control_points)
    generation_result = generate_mechanism_from_curve(
        inference,
        transformed_control_points,
        temperature=temperature,
        mech_type=candidate.mechanism_type_prefix,
        point_prefixes=candidate.point_prefixes,
    )

    mechanism_type = generation_result.get('type')
    mechanism_params = generation_result.get('params', [])
    if not generation_result.get('success') or not mechanism_type or len(mechanism_params) < 3:
        return None

    simulation = simulate_mechanism(mechanism_params, mechanism_type)
    if simulation is None:
        return None

    coupler_traj_original = candidate.restore_coupler_trajectory(simulation.coupler_trajectory)
    transformed_target_curve = candidate.transform_target_curve(target_curve)
    dtw_distance = compute_dtw_distance(target_curve, coupler_traj_original)
    full_index, point, length, angle = extract_constrained_point_metrics(
        mechanism_type,
        mechanism_params,
        candidate.point_prefixes,
    )

    return SearchResult(
        candidate=candidate,
        target_curve=target_curve,
        transformed_target_curve=transformed_target_curve,
        mechanism_type=mechanism_type,
        mechanism_params=mechanism_params,
        coupler_trajectory=coupler_traj_original,
        transformed_coupler_trajectory=simulation.coupler_trajectory,
        wrapper=simulation.wrapper,
        bar_type=simulation.bar_type,
        coords_string=simulation.coords_string,
        dtw_distance=dtw_distance,
        constrained_full_point_index=full_index,
        constrained_point=point,
        constrained_length=length,
        constrained_angle=angle,
    )



def search_sampling_space(inference, control_points, sample, strategy, temperature=0.001, verbose=True):
    """Search a modular sampling space and return the best valid mechanism."""
    control_points = np.asarray(control_points, dtype=np.float32)
    target_curve = reconstruct_curve_from_control_points(control_points)
    if target_curve is None:
        return None

    candidates = list(strategy.build_candidates(sample))
    total_attempts = len(candidates)
    if total_attempts == 0:
        raise ValueError(f'Sampling strategy {strategy.name} produced no candidates.')

    best_result = None
    successful = 0
    progress_every = 1 if total_attempts <= 10 else 5

    if verbose:
        print(
            f"Searching {strategy.name}: {strategy.describe()} = {total_attempts} candidates"
        )

    start_time = time.time()
    for attempt, candidate in enumerate(candidates, start=1):
        try:
            result = evaluate_sampling_candidate(
                inference,
                control_points,
                target_curve,
                candidate,
                temperature,
            )
        except Exception as exc:
            if verbose:
                print(f"  [{attempt}/{total_attempts}] {candidate.label} -> error: {exc}")
            continue

        if result is None:
            continue

        successful += 1
        should_log = verbose and (
            attempt == 1
            or attempt % progress_every == 0
            or best_result is None
            or result.dtw_distance < best_result.dtw_distance
        )
        if should_log:
            print(
                f"  [{attempt}/{total_attempts}] {candidate.label} -> "
                f"{result.mechanism_type}, DTW={result.dtw_distance:.4f}"
            )

        if best_result is None or result.dtw_distance < best_result.dtw_distance:
            best_result = result

    elapsed = time.time() - start_time
    if verbose:
        print(f"\nSearch complete in {elapsed:.2f}s")
        print(f"  Successful simulations: {successful}/{total_attempts}")
        if best_result is not None:
            print(f"  Best candidate: {best_result.candidate.label}")
            print(f"  Best DTW: {best_result.dtw_distance:.4f}")

    return best_result


# ============================================================================
# L-BFGS OPTIMIZATION
# ============================================================================

def optimization_routine(dtw_distance, coupler_traj, coords, b_spline_trajectory, mean, rms_var, fixed_point_indices=None):
    """Optimize mechanism coordinates with L-BFGS-B while respecting fixed points."""
    parts = coords.strip('_').split('_')
    mech_type = parts[-1]
    all_coords = np.array([float(value) for value in parts[:-1]], dtype=np.float64)

    if fixed_point_indices is None:
        fixed_point_indices = fixed_point_indices_for_optimization(mech_type)

    fixed_coord_indices = sorted(
        coord_index
        for point_index in fixed_point_indices
        for coord_index in (2 * point_index, 2 * point_index + 1)
        if coord_index < len(all_coords)
    )
    variable_indices = [index for index in range(len(all_coords)) if index not in fixed_coord_indices]
    coord_values = all_coords[variable_indices]

    if len(variable_indices) == 0:
        wrapper = MechanismWrapper(coords)
        wrapper.simulate()
        final_traj = np.array(wrapper.get_coupler_trajectory())
        final_traj_norm = apply_normalization(final_traj, mean, rms_var)
        _, final_dtw = dtw_path(b_spline_trajectory, final_traj_norm)
        return coords, float(final_dtw), final_traj, b_spline_trajectory

    def objective_function(x_values):
        current_full_coords = all_coords.copy()
        current_full_coords[variable_indices] = x_values
        current_coords = '_' + '_'.join(f"{value:.3f}" for value in current_full_coords) + f'_{mech_type}'

        wrapper = MechanismWrapper(current_coords)
        wrapper.simulate()
        current_traj = np.array(wrapper.get_coupler_trajectory())
        if len(current_traj) == 0:
            return 1e6

        current_traj_norm = apply_normalization(current_traj, mean, rms_var)
        _, current_dtw = dtw_path(b_spline_trajectory, current_traj_norm)
        return float(current_dtw)

    bounds = []
    for value in coord_values:
        bound_range = max(abs(value) * 0.5, 0.5)
        bounds.append((value - bound_range, value + bound_range))

    result = minimize(
        objective_function,
        coord_values,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 50,
            'maxfun': 100,
            'ftol': 1e-6,
            'gtol': 1e-6,
            'eps': 1e-3,
            'disp': False,
        },
    )

    final_full_coords = all_coords.copy()
    final_full_coords[variable_indices] = result.x
    final_coords = '_' + '_'.join(f"{value:.3f}" for value in final_full_coords) + f'_{mech_type}'

    wrapper = MechanismWrapper(final_coords)
    wrapper.simulate()
    final_traj = np.array(wrapper.get_coupler_trajectory())
    final_traj_norm = apply_normalization(final_traj, mean, rms_var)
    _, final_dtw = dtw_path(b_spline_trajectory, final_traj_norm)

    return final_coords, float(final_dtw), final_traj, b_spline_trajectory



def run_lbfgs_optimization(result, verbose=True):
    """Optionally refine the best sampled mechanism while keeping constrained points fixed."""
    normalized_coupler, normalized_target, mean, rms_var = normalize_curves(
        result.transformed_coupler_trajectory,
        result.transformed_target_curve,
    )
    _, initial_dtw = dtw_path(normalized_target, normalized_coupler)

    fixed_points = fixed_point_indices_for_optimization(
        result.mechanism_type,
        result.candidate.point_prefixes,
    )

    if verbose:
        print("\n" + '=' * 60)
        print('L-BFGS OPTIMIZATION')
        print('=' * 60)
        print(f"  Initial DTW (normalized): {initial_dtw:.4f}")
        print(f"  Fixed point indices: {fixed_points}")

    final_coords, final_dtw, final_traj, _ = optimization_routine(
        initial_dtw,
        normalized_coupler,
        result.coords_string,
        normalized_target,
        mean,
        rms_var,
        fixed_point_indices=fixed_points,
    )

    improvement = float(initial_dtw - final_dtw)
    result.optimized_coords = final_coords
    result.optimized_dtw = float(final_dtw)
    result.pre_optimization_dtw = float(initial_dtw)

    if verbose:
        print(f"  Final DTW (normalized): {final_dtw:.4f}")
        print(f"  Improvement: {improvement:.4f}")

    if improvement <= 0:
        if verbose:
            print('  No improvement from optimization')
        return result

    optimized_traj_original = result.candidate.restore_coupler_trajectory(final_traj)
    result.optimized_coupler_trajectory = optimized_traj_original
    result.optimized_dtw_original = compute_dtw_distance(result.target_curve, optimized_traj_original)

    wrapper_optimized = MechanismWrapper(final_coords)
    wrapper_optimized.simulate(steps=200)
    result.wrapper_optimized = wrapper_optimized

    if verbose:
        print(
            f"  Original space DTW: {result.dtw_distance:.4f} -> "
            f"{result.optimized_dtw_original:.4f}"
        )

    return result


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comparison(target_curve, coupler_trajectory, title='Mechanism vs Target', save_path=None):
    """Plot target curve vs mechanism output."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(target_curve[:, 0], target_curve[:, 1], 'r-', linewidth=2, label='Target Curve', alpha=0.8)
    ax.plot(coupler_trajectory[:, 0], coupler_trajectory[:, 1], 'b--', linewidth=2, label='Mechanism Output', alpha=0.8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
        plt.close()
        return

    plt.show()
    plt.close()



def save_optimization_comparison(result, sample_index, save_path):
    """Save a side-by-side plot showing the pre/post optimization result."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(result.target_curve[:, 0], result.target_curve[:, 1], 'r-', linewidth=2, label='Target Curve', alpha=0.8)
    axes[0].plot(result.coupler_trajectory[:, 0], result.coupler_trajectory[:, 1], 'b--', linewidth=2, label='Before Optimization', alpha=0.8)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title(f"Before L-BFGS (DTW={result.dtw_distance:.4f})")

    axes[1].plot(result.target_curve[:, 0], result.target_curve[:, 1], 'r-', linewidth=2, label='Target Curve', alpha=0.8)
    axes[1].plot(result.optimized_coupler_trajectory[:, 0], result.optimized_coupler_trajectory[:, 1], 'g--', linewidth=2, label='After Optimization', alpha=0.8)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_title(f"After L-BFGS (DTW={result.optimized_dtw_original:.4f})")

    plt.suptitle(
        f"Sample {sample_index}: {result.mechanism_type}\n"
        f"Best candidate: {result.candidate.label}"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {save_path}")
    plt.close()



def transform_poses_back(poses, transform_spec):
    """Map poses from mechanism space back to original target-curve space."""
    transformed_poses = []
    for pose in poses:
        transformed_poses.append(transform_spec.invert(np.asarray(pose, dtype=np.float32)))
    return transformed_poses



def create_animation(wrapper, animator, target_curve, coupler_traj, poses, save_path=None):
    """Create an animation showing the mechanism tracing the target curve."""
    fig, ax = plt.subplots(figsize=(12, 8))

    all_x, all_y = [], []
    all_x.extend(target_curve[:, 0])
    all_y.extend(target_curve[:, 1])
    all_x.extend(coupler_traj[:, 0])
    all_y.extend(coupler_traj[:, 1])

    for pose in poses:
        for point in pose:
            all_x.append(point[0])
            all_y.append(point[1])

    margin = 1.0
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin

    def animate_frame(frame):
        ax.clear()

        if frame < len(poses):
            animator.drawer.draw_mechanism(ax, poses[frame], wrapper.get_mechanism_info())

        ax.plot(target_curve[:, 0], target_curve[:, 1], 'r:', linewidth=2, alpha=0.8, label='Target Curve')
        ax.plot(coupler_traj[:, 0], coupler_traj[:, 1], 'b--', linewidth=2, alpha=0.3, label='Mechanism Path')

        if frame > 0:
            ax.plot(coupler_traj[:frame + 1, 0], coupler_traj[:frame + 1, 1], 'b-', linewidth=3, alpha=0.8)

        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend(loc='upper right')
        ax.set_title(f'Frame {frame + 1}/{len(poses)}')

    anim = animation.FuncAnimation(fig, animate_frame, frames=len(poses), interval=50, blit=False, repeat=True)

    if save_path:
        writer = animation.FFMpegWriter(fps=30, bitrate=3000, codec='h264')
        anim.save(save_path, writer=writer, dpi=150)
        print(f"Saved animation to: {save_path}")
        plt.close()
    else:
        plt.show()

    return anim


# ============================================================================
# WORKFLOW ORCHESTRATION
# ============================================================================

@contextmanager
def project_root_cwd():
    """Temporarily chdir to the project root for relative-path-sensitive code."""
    original_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        yield
    finally:
        os.chdir(original_cwd)



def load_inference_model():
    """Load the trained model from the project root so relative assets resolve."""
    with project_root_cwd():
        return MechanismInference(Config.MODEL_SAVE_PATH, Config.VOCAB_SAVE_PATH)



def print_result_summary(result):
    """Print a detailed summary of the best sampled mechanism."""
    print('\n' + '=' * 60)
    print('SEARCH RESULTS')
    print('=' * 60)
    print(f"Best candidate: {result.candidate.label}")
    print(f"  Mechanism Type: {result.mechanism_type} ({result.bar_type})")
    print(f"  DTW Distance: {result.dtw_distance:.4f}")

    if result.candidate.metadata.get('mode') == 'affine':
        tx, ty = result.candidate.metadata['translation']
        print(f"  Rotation: {result.candidate.metadata['angle']:.1f}°")
        print(f"  Translation: ({tx:.2f}, {ty:.2f})")

    if result.candidate.metadata.get('mode') == 'crank-length':
        sampled_x, sampled_y = result.candidate.metadata['sampled_point']
        print(f"  Forced mechanism type prefix: {result.candidate.metadata['mechanism_type_prefix']}")
        print(
            f"  Requested crank sample: length={result.candidate.metadata['length']:.3f}, "
            f"theta={result.candidate.metadata['angle']:.1f}°, point=({sampled_x:.3f}, {sampled_y:.3f})"
        )
        if result.constrained_point is not None:
            print(
                f"  Generated first moving joint (full point {result.constrained_full_point_index}): "
                f"({result.constrained_point[0]:.3f}, {result.constrained_point[1]:.3f})"
            )
            print(
                f"  Generated crank metrics: length={result.constrained_length:.3f}, "
                f"theta={result.constrained_angle:.1f}°"
            )



def save_search_artifacts(result, sample_index, output_dir):
    """Save plots and animation for the best result."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_name = result.candidate.metadata.get('mode', 'sampling')
    plot_path = output_dir / f'comparison_{mode_name}_sample_{sample_index}.png'
    anim_path = output_dir / f'animation_{mode_name}_sample_{sample_index}.mp4'

    if result.optimized_coupler_trajectory is not None:
        save_optimization_comparison(result, sample_index, plot_path)
        wrapper_for_animation = result.wrapper_optimized
        coupler_traj_for_animation = result.optimized_coupler_trajectory
    else:
        plot_comparison(
            result.target_curve,
            result.coupler_trajectory,
            title=(
                f"Sample {sample_index}: {result.mechanism_type} "
                f"(DTW={result.dtw_distance:.4f})\n{result.candidate.label}"
            ),
            save_path=plot_path,
        )
        wrapper_for_animation = result.wrapper
        coupler_traj_for_animation = result.coupler_trajectory

    original_space_poses = transform_poses_back(wrapper_for_animation.poses, result.candidate.transform)
    animator = MechanismAnimator()
    create_animation(
        wrapper_for_animation,
        animator,
        result.target_curve,
        coupler_traj_for_animation,
        original_space_poses,
        save_path=anim_path,
    )



def run_sampling_demo(
    sample_index=0,
    output_dir=DEFAULT_OUTPUT_DIR,
    run_optimization=True,
    strategy=None,
    processed_data_path=None,
    temperature=0.001,
    verbose=True,
):
    """Run a single-sample motion synthesis search with a configurable strategy."""
    if strategy is None:
        strategy = AffineSamplingStrategy()

    print('=' * 60)
    print('MOTION SYNTHESIS WITH MODULAR SAMPLING')
    print('=' * 60)
    print(f"Strategy: {strategy.name} ({strategy.describe()})")

    data = load_processed_data(processed_data_path)
    _, val_indices = create_validation_split(data)
    print(f"Validation set: {len(val_indices)} samples")

    if sample_index >= len(val_indices):
        print(f"Error: sample_index {sample_index} out of range (max: {len(val_indices) - 1})")
        return None

    sample = get_sample(data, val_indices, sample_index)
    control_points = np.asarray(sample['control_points'], dtype=np.float32)

    print(f"\nSample {sample_index}:")
    print(f"  Mechanism type: {sample['mechanism_type']}")
    print(f"  Control points: {len(control_points)}")

    print('\nLoading model...')
    inference = load_inference_model()

    print(f"\nSearching {strategy.name} candidates...")
    result = search_sampling_space(
        inference,
        control_points,
        sample,
        strategy,
        temperature=temperature,
        verbose=verbose,
    )
    if result is None:
        print('\nNo valid mechanism found.')
        return None

    print_result_summary(result)

    if run_optimization:
        run_lbfgs_optimization(result, verbose=verbose)

    save_search_artifacts(result, sample_index, output_dir)

    print('\n' + '=' * 60)
    print('FINAL SUMMARY')
    print('=' * 60)
    print(f"  Mechanism Type: {result.mechanism_type}")
    print(f"  Search DTW: {result.dtw_distance:.4f}")
    if result.optimized_dtw_original is not None:
        improvement = result.dtw_distance - result.optimized_dtw_original
        print(f"  After L-BFGS DTW: {result.optimized_dtw_original:.4f}")
        print(f"  Total Improvement: {improvement:.4f} ({100 * improvement / result.dtw_distance:.1f}%)")
    print(f"  Output directory: {output_dir}")

    return result



def run_affine_search_demo(sample_index=0, output_dir=DEFAULT_OUTPUT_DIR, run_optimization=True, processed_data_path=None, temperature=0.001, angles=None, translations=None, verbose=True):
    """Backward-compatible wrapper for the original affine search workflow."""
    strategy = AffineSamplingStrategy(
        angles=list(DEFAULT_AFFINE_ANGLES if angles is None else angles),
        translations=list(DEFAULT_AFFINE_TRANSLATIONS if translations is None else translations),
    )
    return run_sampling_demo(
        sample_index=sample_index,
        output_dir=output_dir,
        run_optimization=run_optimization,
        strategy=strategy,
        processed_data_path=processed_data_path,
        temperature=temperature,
        verbose=verbose,
    )



def run_crank_length_search_demo(sample_index=0, output_dir=DEFAULT_OUTPUT_DIR, crank_lengths=None, crank_angles=None, mechanism_type=None, run_optimization=True, processed_data_path=None, temperature=0.001, verbose=True):
    """Search by prefixed crank-point locations sampled on specified circles."""
    strategy = CrankLengthSamplingStrategy(
        lengths=list(DEFAULT_CRANK_LENGTHS if crank_lengths is None else crank_lengths),
        crank_angles=list(build_circle_angles(DEFAULT_CRANK_ANGLE_STEP) if crank_angles is None else crank_angles),
        mechanism_type=mechanism_type,
    )
    return run_sampling_demo(
        sample_index=sample_index,
        output_dir=output_dir,
        run_optimization=run_optimization,
        strategy=strategy,
        processed_data_path=processed_data_path,
        temperature=temperature,
        verbose=verbose,
    )


# ============================================================================
# CLI
# ============================================================================

def parse_translation_pairs(values):
    """Parse comma-separated translation pairs from the CLI."""
    if not values:
        return list(DEFAULT_AFFINE_TRANSLATIONS)

    translations = []
    for value in values:
        parts = value.split(',')
        if len(parts) != 2:
            raise ValueError(
                f"Invalid translation '{value}'. Expected comma-separated tx,ty pairs."
            )
        translations.append((float(parts[0]), float(parts[1])))
    return translations



def build_sampling_strategy_from_args(args):
    """Instantiate the requested sampling strategy from CLI arguments."""
    if args.sampling_mode == 'affine':
        angles = list(DEFAULT_AFFINE_ANGLES if args.affine_angles is None else args.affine_angles)
        translations = parse_translation_pairs(args.affine_translations)
        return AffineSamplingStrategy(angles=angles, translations=translations)

    crank_lengths = list(DEFAULT_CRANK_LENGTHS if args.crank_lengths is None else args.crank_lengths)
    crank_angles = list(build_circle_angles(args.crank_angle_step) if args.crank_angles is None else args.crank_angles)
    return CrankLengthSamplingStrategy(
        lengths=crank_lengths,
        crank_angles=crank_angles,
        mechanism_type=args.mechanism_type,
    )



def build_arg_parser():
    parser = argparse.ArgumentParser(description='Motion synthesis with modular sampling strategies.')
    parser.add_argument('--sample-index', type=int, default=84636, help='Validation sample index to evaluate.')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR), help='Directory for plots and animations.')
    parser.add_argument('--processed-data-path', type=str, default=None, help='Optional processed_data.pkl path override.')
    parser.add_argument('--temperature', type=float, default=0.001, help='Sampling temperature for autoregressive generation.')
    parser.add_argument('--sampling-mode', choices=['affine', 'crank-length'], default='affine', help='Sampling strategy to use.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--quiet', action='store_true', help='Reduce progress logging.')
    parser.add_argument('--no-optimization', action='store_true', help='Skip the L-BFGS post-optimization stage.')

    parser.add_argument('--affine-angles', type=float, nargs='*', default=None, help='Angles to test for affine sampling.')
    parser.add_argument(
        '--affine-translations',
        nargs='*',
        default=None,
        help='Translation pairs as tx,ty entries (for example: 0,0 0.3,0 0,0.3).',
    )

    parser.add_argument('--crank-lengths', type=float, nargs='*', default=None, help='Normalized crank lengths to sample.')
    parser.add_argument('--crank-angles', type=float, nargs='*', default=None, help='Explicit crank angles in degrees.')
    parser.add_argument('--crank-angle-step', type=float, default=DEFAULT_CRANK_ANGLE_STEP, help='Angle step used when crank angles are not provided.')
    parser.add_argument('--mechanism-type', type=str, default=None, help='Optional mechanism-type prefix for crank-length sampling. Defaults to the sample type.')
    return parser



def set_random_seeds(seed):
    """Set reproducible seeds for Python, NumPy, and Torch."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def main():
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    set_random_seeds(args.seed)
    strategy = build_sampling_strategy_from_args(args)
    result = run_sampling_demo(
        sample_index=args.sample_index,
        output_dir=args.output_dir,
        run_optimization=not args.no_optimization,
        strategy=strategy,
        processed_data_path=args.processed_data_path,
        temperature=args.temperature,
        verbose=not args.quiet,
    )

    if result:
        print(f"\nSuccess! Check output in: {args.output_dir}")
    else:
        print('\nDemo failed.')


if __name__ == '__main__':
    main()
