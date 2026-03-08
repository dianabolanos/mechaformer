from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_synthesis.single_mechanism_with_sampling import (
    DEFAULT_CRANK_ANGLE_STEP,
    DEFAULT_CRANK_LENGTHS,
    AffineSamplingStrategy,
    CrankLengthSamplingStrategy,
    SearchResult,
    SamplingStrategy,
    build_circle_angles,
    build_sampling_strategy_from_args,
    create_validation_split,
    evaluate_sampling_candidate,
    get_sample,
    load_inference_model,
    load_processed_data,
    reconstruct_curve_from_control_points,
    save_search_artifacts,
    set_random_seeds,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'smoothness_output'


@dataclass(frozen=True)
class SelectionWeights:
    curve_following: float
    smoothness: float

    @classmethod
    def from_mode(cls, mode):
        presets = {
            'balanced': cls(curve_following=0.7, smoothness=0.3),
            'curve': cls(curve_following=0.85, smoothness=0.15),
            'smoothness': cls(curve_following=0.35, smoothness=0.65),
        }
        if mode not in presets:
            raise ValueError(f"Unknown optimize_for mode '{mode}'.")
        return presets[mode]

    def normalized(self):
        total = self.curve_following + self.smoothness
        if total <= 0:
            raise ValueError('Selection weights must sum to a positive value.')
        return SelectionWeights(
            curve_following=self.curve_following / total,
            smoothness=self.smoothness / total,
        )


@dataclass(frozen=True)
class MotionSmoothnessMetrics:
    mean_speed: float
    peak_speed: float
    speed_variation: float
    jerk_rms: float
    smoothness_loss: float


@dataclass(frozen=True)
class RankedCandidate:
    result: SearchResult
    smoothness_metrics: MotionSmoothnessMetrics
    normalized_curve_loss: float
    normalized_smoothness_loss: float
    selection_score: float


@dataclass
class SmoothnessSelectionRun:
    sample_index: int
    sample_mechanism_type: str
    strategy_name: str
    weights: SelectionWeights
    ranked_candidates: list[RankedCandidate]
    report_paths: dict[str, Path]
    selected_visualization_dir: Optional[Path] = None

    @property
    def selected_candidate(self):
        return self.ranked_candidates[0]


# ============================================================================
# METRICS
# ============================================================================

def compute_motion_smoothness(coupler_trajectory):
    """Measure how fast and jerky the end-effector motion is.

    The metric assumes a uniform timestep between simulation poses and combines:
    - normalized mean speed,
    - normalized peak speed,
    - relative speed variation,
    - normalized RMS jerk.

    Lower values are smoother.
    """
    trajectory = np.asarray(coupler_trajectory, dtype=np.float64)
    if len(trajectory) < 4:
        return MotionSmoothnessMetrics(
            mean_speed=float('inf'),
            peak_speed=float('inf'),
            speed_variation=float('inf'),
            jerk_rms=float('inf'),
            smoothness_loss=float('inf'),
        )

    velocity = np.diff(trajectory, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)

    trajectory_extent = max(np.linalg.norm(np.ptp(trajectory, axis=0)), 1e-6)
    mean_speed = float(np.mean(speed) / trajectory_extent)
    peak_speed = float(np.max(speed) / trajectory_extent)
    speed_variation = float(np.std(speed) / (np.mean(speed) + 1e-6))
    jerk_rms = float(np.sqrt(np.mean(np.sum(jerk ** 2, axis=1))) / trajectory_extent)

    smoothness_loss = (
        0.30 * mean_speed
        + 0.20 * peak_speed
        + 0.20 * speed_variation
        + 0.30 * jerk_rms
    )
    return MotionSmoothnessMetrics(
        mean_speed=mean_speed,
        peak_speed=peak_speed,
        speed_variation=speed_variation,
        jerk_rms=jerk_rms,
        smoothness_loss=float(smoothness_loss),
    )



def resolve_selection_weights(optimize_for='balanced', curve_weight=None, smoothness_weight=None):
    """Resolve preset or custom weights for ranking."""
    if curve_weight is None and smoothness_weight is None:
        return SelectionWeights.from_mode(optimize_for).normalized()
    if curve_weight is None or smoothness_weight is None:
        raise ValueError('curve_weight and smoothness_weight must be provided together.')
    return SelectionWeights(curve_following=curve_weight, smoothness=smoothness_weight).normalized()


# ============================================================================
# CANDIDATE COLLECTION AND RANKING
# ============================================================================

def collect_candidate_results(inference, control_points, sample, strategy, temperature=0.001, max_candidates=None, verbose=True):
    """Evaluate all valid sampled mechanisms for a strategy."""
    target_curve = reconstruct_curve_from_control_points(control_points)
    if target_curve is None:
        return []

    candidates = list(strategy.build_candidates(sample))
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    valid_results = []
    total_candidates = len(candidates)
    progress_every = 1 if total_candidates <= 10 else 5

    if verbose:
        print(f"Evaluating {total_candidates} candidate mechanisms for strategy '{strategy.name}'")

    for index, candidate in enumerate(candidates, start=1):
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
                print(f"  [{index}/{total_candidates}] {candidate.label} -> error: {exc}")
            continue

        if result is None:
            continue

        valid_results.append(result)
        if verbose and (index == 1 or index % progress_every == 0):
            print(
                f"  [{index}/{total_candidates}] {candidate.label} -> "
                f"{result.mechanism_type}, DTW={result.dtw_distance:.4f}"
            )

    if verbose:
        print(f"Valid mechanisms collected: {len(valid_results)}/{total_candidates}")

    return valid_results



def _normalize_values(values):
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if np.isclose(min_value, max_value):
        return [0.0 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]



def rank_candidate_results(results, weights):
    """Rank valid mechanisms by weighted curve-following and smoothness losses."""
    if not results:
        return []

    weights = weights.normalized()
    smoothness_metrics = [compute_motion_smoothness(result.coupler_trajectory) for result in results]
    normalized_curve_losses = _normalize_values([result.dtw_distance for result in results])
    normalized_smoothness_losses = _normalize_values(
        [metrics.smoothness_loss for metrics in smoothness_metrics]
    )

    ranked_candidates = []
    for result, metrics, curve_loss, smoothness_loss in zip(
        results,
        smoothness_metrics,
        normalized_curve_losses,
        normalized_smoothness_losses,
    ):
        selection_score = (
            weights.curve_following * curve_loss
            + weights.smoothness * smoothness_loss
        )
        ranked_candidates.append(
            RankedCandidate(
                result=result,
                smoothness_metrics=metrics,
                normalized_curve_loss=curve_loss,
                normalized_smoothness_loss=smoothness_loss,
                selection_score=float(selection_score),
            )
        )

    ranked_candidates.sort(key=lambda candidate: candidate.selection_score)
    return ranked_candidates


# ============================================================================
# REPORTING
# ============================================================================

def _truncate(text, max_length=26):
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + '...'


def _is_crank_length_mode(ranked_candidates):
    return bool(ranked_candidates) and (
        ranked_candidates[0].result.candidate.metadata.get('mode') == 'crank-length'
    )



def format_ranked_results_table(ranked_candidates, top_rows=None):
    """Format a readable table for the ranked candidate mechanisms."""
    candidates_to_show = ranked_candidates if top_rows is None else ranked_candidates[:top_rows]
    crank_mode = _is_crank_length_mode(candidates_to_show)
    if crank_mode:
        headers = [
            'Rank',
            'Candidate',
            'Type',
            'ReqLen',
            'GenLen',
            'ReqAng',
            'GenAng',
            'DTW',
            'Smooth',
            'Score',
            'Pick',
        ]
    else:
        headers = ['Rank', 'Candidate', 'Type', 'DTW', 'Smooth', 'MeanSpd', 'Jerk', 'Score', 'Pick']
    rows = []
    for rank, candidate in enumerate(candidates_to_show, start=1):
        if crank_mode:
            requested_length = candidate.result.candidate.metadata.get('length')
            requested_angle = candidate.result.candidate.metadata.get('angle')
            rows.append([
                str(rank),
                _truncate(candidate.result.candidate.label),
                candidate.result.mechanism_type,
                f"{requested_length:.3f}" if requested_length is not None else 'n/a',
                f"{candidate.result.constrained_length:.3f}" if candidate.result.constrained_length is not None else 'n/a',
                f"{requested_angle:.1f}" if requested_angle is not None else 'n/a',
                f"{candidate.result.constrained_angle:.1f}" if candidate.result.constrained_angle is not None else 'n/a',
                f"{candidate.result.dtw_distance:.4f}",
                f"{candidate.smoothness_metrics.smoothness_loss:.4f}",
                f"{candidate.selection_score:.4f}",
                '<--' if rank == 1 else '',
            ])
        else:
            rows.append([
                str(rank),
                _truncate(candidate.result.candidate.label),
                candidate.result.mechanism_type,
                f"{candidate.result.dtw_distance:.4f}",
                f"{candidate.smoothness_metrics.smoothness_loss:.4f}",
                f"{candidate.smoothness_metrics.mean_speed:.4f}",
                f"{candidate.smoothness_metrics.jerk_rms:.4f}",
                f"{candidate.selection_score:.4f}",
                '<--' if rank == 1 else '',
            ])

    widths = [
        max(len(headers[column_index]), max((len(row[column_index]) for row in rows), default=0))
        for column_index in range(len(headers))
    ]
    separator = '-+-'.join('-' * width for width in widths)
    header_line = ' | '.join(header.ljust(widths[index]) for index, header in enumerate(headers))
    row_lines = [
        ' | '.join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    ]
    return '\n'.join([header_line, separator, *row_lines])



def build_selected_summary(selected_candidate, weights):
    result = selected_candidate.result
    metrics = selected_candidate.smoothness_metrics
    lines = [
        f"Selected candidate: {result.candidate.label}",
        f"  Mechanism type: {result.mechanism_type} ({result.bar_type})",
        f"  Curve DTW: {result.dtw_distance:.4f}",
        f"  Smoothness loss: {metrics.smoothness_loss:.4f}",
        f"  Mean speed: {metrics.mean_speed:.4f}",
        f"  Peak speed: {metrics.peak_speed:.4f}",
        f"  Speed variation: {metrics.speed_variation:.4f}",
        f"  Jerk RMS: {metrics.jerk_rms:.4f}",
        f"  Weighted score: {selected_candidate.selection_score:.4f}",
        f"  Weights -> curve: {weights.curve_following:.2f}, smoothness: {weights.smoothness:.2f}",
    ]
    if result.candidate.metadata.get('mode') == 'crank-length':
        requested_length = result.candidate.metadata.get('length')
        requested_angle = result.candidate.metadata.get('angle')
        lines.append(
            f"  Requested crank sample -> length: {requested_length:.3f}, angle: {requested_angle:.1f}°"
        )
        if result.constrained_length is not None:
            lines.append(
                f"  Generated first moving joint -> length: {result.constrained_length:.3f}, "
                f"angle: {result.constrained_angle:.1f}°"
            )
    return '\n'.join(lines)


def save_crank_sampling_overview(ranked_candidates, sample_index, output_dir, top_rows=6):
    """Create a visual overview of requested vs generated crank samples."""
    if not _is_crank_length_mode(ranked_candidates):
        return None

    candidates_to_show = ranked_candidates if top_rows is None else ranked_candidates[:top_rows]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = output_dir / f'crank_sampling_overview_sample_{sample_index}.png'

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(candidates_to_show)))

    crank_ax, trajectory_ax, dtw_ax = axes
    unique_lengths = sorted({
        candidate.result.candidate.metadata.get('length')
        for candidate in candidates_to_show
        if candidate.result.candidate.metadata.get('length') is not None
    })
    for length in unique_lengths:
        crank_ax.add_patch(
            plt.Circle((0.0, 0.0), length, fill=False, linestyle='--', color='gray', alpha=0.25)
        )

    crank_ax.plot([0.0, 1.0], [0.0, 0.0], color='black', linewidth=1.5, alpha=0.4)
    crank_ax.scatter([0.0], [0.0], color='black', s=40)
    crank_ax.scatter([1.0], [0.0], color='black', s=40)
    crank_ax.scatter([], [], marker='x', color='black', label='Requested crank sample')
    crank_ax.scatter([], [], marker='o', color='black', label='Generated first moving joint')

    target_curve = candidates_to_show[0].result.target_curve
    trajectory_ax.plot(
        target_curve[:, 0],
        target_curve[:, 1],
        color='black',
        linewidth=2.5,
        label='Target curve',
        alpha=0.8,
    )

    labels = []
    dtw_values = []
    for index, (candidate, color) in enumerate(zip(candidates_to_show, colors), start=1):
        requested_point = np.asarray(candidate.result.candidate.metadata['sampled_point'], dtype=np.float64)
        generated_point = candidate.result.constrained_point
        crank_ax.scatter(requested_point[0], requested_point[1], marker='x', s=90, color=color)
        if generated_point is not None:
            crank_ax.scatter(
                generated_point[0],
                generated_point[1],
                marker='o',
                s=80,
                color=color,
                edgecolors='black',
                linewidths=0.6,
            )
            crank_ax.plot(
                [requested_point[0], generated_point[0]],
                [requested_point[1], generated_point[1]],
                linestyle='--',
                color=color,
                alpha=0.6,
            )
            crank_ax.text(generated_point[0], generated_point[1], str(index), fontsize=9, weight='bold')

        trajectory_ax.plot(
            candidate.result.coupler_trajectory[:, 0],
            candidate.result.coupler_trajectory[:, 1],
            color=color,
            linewidth=3.0 if index == 1 else 1.8,
            alpha=0.9 if index == 1 else 0.45,
            label=f'#{index} {candidate.result.candidate.label}',
        )
        labels.append(f'#{index}')
        dtw_values.append(candidate.result.dtw_distance)

    bars = dtw_ax.bar(labels, dtw_values, color=colors, alpha=0.85)
    if len(bars) > 0:
        bars[0].set_edgecolor('black')
        bars[0].set_linewidth(2.0)
    for bar, value in zip(bars, dtw_values):
        dtw_ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f'{value:.1f}',
            ha='center',
            va='bottom',
            fontsize=9,
        )

    crank_ax.set_title('Requested vs generated crank points')
    crank_ax.set_xlabel('X')
    crank_ax.set_ylabel('Y')
    crank_ax.set_aspect('equal')
    crank_ax.grid(True, alpha=0.3)
    crank_ax.legend(loc='upper right')

    trajectory_ax.set_title('Target curve vs top candidate trajectories')
    trajectory_ax.set_xlabel('X')
    trajectory_ax.set_ylabel('Y')
    trajectory_ax.set_aspect('equal')
    trajectory_ax.grid(True, alpha=0.3)
    trajectory_ax.legend(loc='best', fontsize=8)

    dtw_ax.set_title('Candidate DTW ranking')
    dtw_ax.set_xlabel('Candidate rank')
    dtw_ax.set_ylabel('DTW distance')
    dtw_ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle(
        f'Crank-length sampling overview (sample {sample_index})\n'
        f'Selected candidate: {candidates_to_show[0].result.candidate.label}',
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(overview_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return overview_path



def _serialize_ranked_candidate(rank, candidate):
    result = candidate.result
    metrics = candidate.smoothness_metrics
    return {
        'rank': rank,
        'candidate_label': result.candidate.label,
        'candidate_metadata': result.candidate.metadata,
        'mechanism_type': result.mechanism_type,
        'bar_type': result.bar_type,
        'dtw_distance': result.dtw_distance,
        'smoothness_loss': metrics.smoothness_loss,
        'mean_speed': metrics.mean_speed,
        'peak_speed': metrics.peak_speed,
        'speed_variation': metrics.speed_variation,
        'jerk_rms': metrics.jerk_rms,
        'normalized_curve_loss': candidate.normalized_curve_loss,
        'normalized_smoothness_loss': candidate.normalized_smoothness_loss,
        'selection_score': candidate.selection_score,
        'coords_string': result.coords_string,
        'mechanism_params': result.mechanism_params,
    }



def save_ranked_reports(ranked_candidates, weights, sample_index, output_dir, top_rows=None):
    """Save a JSON/CSV/text report for all ranked candidates."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = format_ranked_results_table(ranked_candidates, top_rows=top_rows)
    summary = build_selected_summary(ranked_candidates[0], weights)

    text_path = output_dir / f'smoothness_report_sample_{sample_index}.txt'
    json_path = output_dir / f'smoothness_report_sample_{sample_index}.json'
    csv_path = output_dir / f'smoothness_report_sample_{sample_index}.csv'

    text_path.write_text(table + '\n\n' + summary + '\n', encoding='utf-8')

    json_payload = {
        'sample_index': sample_index,
        'weights': asdict(weights),
        'selected_candidate': _serialize_ranked_candidate(1, ranked_candidates[0]),
        'ranked_candidates': [
            _serialize_ranked_candidate(rank, candidate)
            for rank, candidate in enumerate(ranked_candidates, start=1)
        ],
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding='utf-8')

    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_serialize_ranked_candidate(1, ranked_candidates[0]).keys()))
        writer.writeheader()
        for rank, candidate in enumerate(ranked_candidates, start=1):
            writer.writerow(_serialize_ranked_candidate(rank, candidate))

    return {
        'text': text_path,
        'json': json_path,
        'csv': csv_path,
    }


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def run_smoothness_selection_demo(
    sample_index=0,
    output_dir=DEFAULT_OUTPUT_DIR,
    strategy=None,
    optimize_for='balanced',
    curve_weight=None,
    smoothness_weight=None,
    processed_data_path=None,
    temperature=0.001,
    max_candidates=None,
    top_report_rows=10,
    visualize_selected=True,
    verbose=True,
):
    """Sample mechanisms, rank them by smoothness + curve following, and visualize the winner."""
    if strategy is None:
        strategy = AffineSamplingStrategy()
    weights = resolve_selection_weights(
        optimize_for=optimize_for,
        curve_weight=curve_weight,
        smoothness_weight=smoothness_weight,
    )

    print('=' * 60)
    print('SMOOTHNESS-WEIGHTED MOTION SYNTHESIS')
    print('=' * 60)
    print(f"Strategy: {strategy.name} ({strategy.describe()})")
    print(
        f"Selection weights -> curve: {weights.curve_following:.2f}, "
        f"smoothness: {weights.smoothness:.2f}"
    )

    data = load_processed_data(processed_data_path)
    _, val_indices = create_validation_split(data)
    print(f"Validation set: {len(val_indices)} samples")

    if sample_index >= len(val_indices):
        print(f"Error: sample_index {sample_index} out of range (max: {len(val_indices) - 1})")
        return None

    sample = get_sample(data, val_indices, sample_index)
    control_points = np.asarray(sample['control_points'], dtype=np.float32)
    print(f"\nSample {sample_index}: {sample['mechanism_type']}")

    print('\nLoading model...')
    inference = load_inference_model()

    valid_results = collect_candidate_results(
        inference,
        control_points,
        sample,
        strategy,
        temperature=temperature,
        max_candidates=max_candidates,
        verbose=verbose,
    )
    if not valid_results:
        print('\nNo valid mechanisms found for smoothness ranking.')
        return None

    ranked_candidates = rank_candidate_results(valid_results, weights)
    print('\n' + '=' * 60)
    print('CANDIDATE RANKING')
    print('=' * 60)
    print(format_ranked_results_table(ranked_candidates, top_rows=top_report_rows))
    print('\n' + build_selected_summary(ranked_candidates[0], weights))

    report_paths = save_ranked_reports(
        ranked_candidates,
        weights,
        sample_index,
        output_dir,
        top_rows=top_report_rows,
    )
    crank_overview_path = save_crank_sampling_overview(
        ranked_candidates,
        sample_index,
        output_dir,
        top_rows=min(top_report_rows, 6) if top_report_rows is not None else 6,
    )
    if crank_overview_path is not None:
        report_paths['crank_overview'] = crank_overview_path
    print('\nSaved reports:')
    for label, path in report_paths.items():
        print(f"  {label}: {path}")

    selected_visualization_dir = None
    if visualize_selected:
        selected_visualization_dir = Path(output_dir) / 'selected_visualization'
        save_search_artifacts(ranked_candidates[0].result, sample_index, selected_visualization_dir)
        print(f"Selected mechanism visualization saved to: {selected_visualization_dir}")

    return SmoothnessSelectionRun(
        sample_index=sample_index,
        sample_mechanism_type=sample['mechanism_type'],
        strategy_name=strategy.name,
        weights=weights,
        ranked_candidates=ranked_candidates,
        report_paths=report_paths,
        selected_visualization_dir=selected_visualization_dir,
    )


# ============================================================================
# CLI
# ============================================================================

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Rank sampled mechanisms by curve following and end-effector smoothness.')
    parser.add_argument('--sample-index', type=int, default=365, help='Validation sample index to evaluate.')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR), help='Directory for reports and selected-mechanism artifacts.')
    parser.add_argument('--processed-data-path', type=str, default=None, help='Optional processed_data.pkl path override.')
    parser.add_argument('--temperature', type=float, default=0.001, help='Sampling temperature for autoregressive generation.')
    parser.add_argument('--sampling-mode', choices=['affine', 'crank-length'], default='affine', help='Sampling strategy to use before ranking.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--quiet', action='store_true', help='Reduce progress logging.')
    parser.add_argument('--max-candidates', type=int, default=None, help='Optional cap on the number of sampled candidates to score.')
    parser.add_argument('--top-report-rows', type=int, default=10, help='How many ranked rows to print/save in the human-readable table.')
    parser.add_argument('--optimize-for', choices=['balanced', 'curve', 'smoothness'], default='balanced', help='Preset weighting mode for selection.')
    parser.add_argument('--curve-weight', type=float, default=None, help='Optional custom curve-following weight.')
    parser.add_argument('--smoothness-weight', type=float, default=None, help='Optional custom smoothness weight.')
    parser.add_argument('--no-visualize-selected', action='store_true', help='Skip visualization artifacts for the selected mechanism.')

    parser.add_argument('--affine-angles', type=float, nargs='*', default=None, help='Angles to test for affine sampling.')
    parser.add_argument('--affine-translations', nargs='*', default=None, help='Translation pairs as tx,ty entries (for example: 0,0 0.3,0).')
    parser.add_argument('--crank-lengths', type=float, nargs='*', default=None, help='Normalized crank lengths to sample for crank-length mode.')
    parser.add_argument('--crank-angles', type=float, nargs='*', default=None, help='Explicit crank angles in degrees.')
    parser.add_argument('--crank-angle-step', type=float, default=DEFAULT_CRANK_ANGLE_STEP, help='Angle step used when crank angles are not provided.')
    parser.add_argument('--mechanism-type', type=str, default=None, help='Optional mechanism-type prefix for crank-length sampling. Defaults to the sample type.')
    return parser



def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    set_random_seeds(args.seed)
    strategy = build_sampling_strategy_from_args(args)
    result = run_smoothness_selection_demo(
        sample_index=args.sample_index,
        output_dir=args.output_dir,
        strategy=strategy,
        optimize_for=args.optimize_for,
        curve_weight=args.curve_weight,
        smoothness_weight=args.smoothness_weight,
        processed_data_path=args.processed_data_path,
        temperature=args.temperature,
        max_candidates=args.max_candidates,
        top_report_rows=args.top_report_rows,
        visualize_selected=not args.no_visualize_selected,
        verbose=not args.quiet,
    )

    if result is None:
        print('\nNo selection result was produced.')
    else:
        print(f"\nSelected candidate: {result.selected_candidate.result.candidate.label}")


if __name__ == '__main__':
    main()
