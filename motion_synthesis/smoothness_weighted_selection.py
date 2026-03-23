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
    run_lbfgs_optimization,
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
    speed_cv: float           # coefficient of variation: std(speed) / mean(speed)
    normalized_jerk: float    # RMS jerk / mean speed
    composite_loss: float     # (speed_cv + normalized_jerk) / 2
    sparc: float              # Spectral Arc Length (less negative = smoother, ≤ 0)
    smoothness_loss: float    # active ranking metric: composite_loss or -sparc


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

def compute_sparc(speed, padlevel=4, fc=0.1, amp_th=0.05):
    """Spectral Arc Length (SPARC) smoothness metric.

    Measures the arc length of the normalised speed magnitude spectrum over the
    low-frequency region.  Scale-invariant: amplitude normalised by spectrum peak;
    frequency axis normalised by the effective cutoff.

    Reference: Balasubramanian et al., JNER 2012.

    Args:
        speed:     1-D speed profile (non-negative, one value per trajectory step).
        padlevel:  FFT zero-padding factor (default 4 for smoother spectrum).
        fc:        Frequency cutoff as a fraction of Nyquist (0.5 cycles/step).
                   Default 0.1 retains the lowest 10 % of frequencies.
        amp_th:    Amplitude threshold: auto-truncates above the first bin where
                   the normalised spectrum drops below this value (default 0.05).

    Returns:
        float ≤ 0.  Less negative (closer to 0) means smoother motion.
    """
    speed = np.asarray(speed, dtype=np.float64)
    n = len(speed)
    if n < 4:
        return float('-inf')

    speed_max = float(np.max(np.abs(speed)))
    if speed_max < 1e-10:
        return 0.0  # Stationary → perfectly smooth (zero arc length)

    nfft = int(2 ** np.ceil(np.log2(n * padlevel)))
    Mfft = np.abs(np.fft.rfft(speed, n=nfft))
    n_one_sided = len(Mfft)

    # Normalise by spectrum peak
    Mfft_norm = Mfft / (np.max(Mfft) + 1e-10)

    # Frequency axis in cycles/step: f[k] = k / nfft ∈ [0, 0.5]
    f = np.arange(n_one_sided) / nfft
    df = f[1] if n_one_sided > 1 else 1.0

    # Amplitude-based cutoff: first bin below amp_th
    below_th = np.where(Mfft_norm < amp_th)[0]
    fc_th_idx = int(below_th[0]) if len(below_th) > 0 else n_one_sided - 1

    # User-specified frequency cutoff (fraction of Nyquist → absolute index)
    fc_abs = fc * 0.5
    fc_user_idx = int(np.searchsorted(f, fc_abs))

    fc_idx = max(2, min(fc_th_idx, fc_user_idx, n_one_sided - 1))

    # Arc length: -∫₀^ωc √((1/ωc)² + (dV̂/dω)²) dω
    Mf = Mfft_norm[:fc_idx + 1]
    dM = np.diff(Mf) / df                             # dV̂/dω
    fc_norm = 1.0 / max(float(f[fc_idx]), df)         # 1/ωc normalisation
    arc_length = float(np.sum(np.sqrt(fc_norm ** 2 + dM ** 2)) * df)
    return -arc_length


def compute_motion_smoothness(coupler_trajectory, metric='composite'):
    """Speed-invariant measure of end-effector motion smoothness.

    Computes both the composite loss and SPARC.  ``smoothness_loss`` is set to
    the active metric chosen by ``metric``.

    Composite components (equal weight = 1/2 each):
      speed_cv        - coefficient of variation std(v) / mean(v)
      normalized_jerk  - RMS(jerk) / mean(v)

    SPARC (Spectral Arc Length):
      Arc length of the normalised speed spectrum in the low-frequency region.
      Stored as a non-positive value; ``smoothness_loss = -sparc`` when
      ``metric='sparc'`` so that lower is still better.

    Lower ``smoothness_loss`` always indicates smoother motion.
    """
    trajectory = np.asarray(coupler_trajectory, dtype=np.float64)
    if len(trajectory) < 4:
        return MotionSmoothnessMetrics(
            speed_cv=float('inf'),
            normalized_jerk=float('inf'),
            composite_loss=float('inf'),
            sparc=float('-inf'),
            smoothness_loss=float('inf'),
        )

    velocity = np.diff(trajectory, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)

    mean_spd = float(np.mean(speed))
    if mean_spd < 1e-9:
        # Essentially stationary → perfectly smooth by all metrics.
        return MotionSmoothnessMetrics(
            speed_cv=0.0,
            normalized_jerk=0.0,
            composite_loss=0.0,
            sparc=0.0,
            smoothness_loss=0.0,
        )

    speed_cv = float(np.std(speed) / mean_spd)
    jerk_magnitude = np.sqrt(np.sum(jerk ** 2, axis=1))
    normalized_jerk = float(np.sqrt(np.mean(jerk_magnitude ** 2)) / mean_spd)
    composite_loss = (speed_cv + normalized_jerk) / 2.0

    sparc_val = compute_sparc(speed)
    sparc_loss = -sparc_val  # positive, lower = smoother

    smoothness_loss = composite_loss if metric == 'composite' else sparc_loss

    return MotionSmoothnessMetrics(
        speed_cv=speed_cv,
        normalized_jerk=normalized_jerk,
        composite_loss=float(composite_loss),
        sparc=float(sparc_val),
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



def rank_candidate_results(results, weights, smoothness_metric='composite'):
    """Rank valid mechanisms by weighted curve-following and smoothness losses."""
    if not results:
        return []

    weights = weights.normalized()
    smoothness_metrics = [
        compute_motion_smoothness(result.coupler_trajectory, metric=smoothness_metric)
        for result in results
    ]
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
        headers = ['Rank', 'Candidate', 'Type', 'DTW', 'Smooth', 'SpeedCV', 'NormJerk', 'Score', 'Pick']
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
                f"{candidate.smoothness_metrics.speed_cv:.4f}",
                f"{candidate.smoothness_metrics.normalized_jerk:.4f}",
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
        f"  Smoothness loss (active): {metrics.smoothness_loss:.4f}",
        f"  -- Composite loss: {metrics.composite_loss:.4f}",
        f"     Speed CV (std/mean): {metrics.speed_cv:.4f}",
        f"     Normalized jerk (RMS/mean): {metrics.normalized_jerk:.4f}",
        f"  -- SPARC: {metrics.sparc:.4f}  (loss = {-metrics.sparc:.4f})",
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
        'composite_loss': metrics.composite_loss,
        'sparc': metrics.sparc,
        'speed_cv': metrics.speed_cv,
        'normalized_jerk': metrics.normalized_jerk,
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
# PARETO FRONT VISUALIZATION
# ============================================================================

def _compute_pareto_front(dtw_values, smoothness_values):
    """Return boolean mask of non-dominated (Pareto-optimal) candidates.

    A candidate is Pareto-optimal if no other candidate is strictly better on
    *both* DTW distance (lower is better) and smoothness_loss (lower is better).
    """
    n = len(dtw_values)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dtw_values[j] <= dtw_values[i] and smoothness_values[j] <= smoothness_values[i]:
                if dtw_values[j] < dtw_values[i] or smoothness_values[j] < smoothness_values[i]:
                    is_pareto[i] = False
                    break
    return is_pareto


def plot_pareto_front(ranked_candidates, sample_index, output_dir, weights=None, smoothness_metric='composite'):
    """Scatter-plot DTW vs smoothness_loss for all candidates and draw the Pareto front.

    Background is shaded by weighted score (blue = good, red = poor) with iso-score
    contour lines showing trade-off trends.  The Pareto-optimal subset is highlighted
    and the top-ranked candidate is starred.

    Args:
        ranked_candidates: List of RankedCandidate objects (already ranked).
        sample_index: Integer sample index (used in the filename and title).
        output_dir: Path-like directory where the PNG is saved.
        weights: Optional SelectionWeights used to label the selection mode.

    Returns:
        Path to the saved PNG, or None if plotting failed.
    """
    if not ranked_candidates:
        return None

    from matplotlib.colors import LinearSegmentedColormap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtw_vals = np.array([c.result.dtw_distance for c in ranked_candidates], dtype=float)
    smooth_vals = np.array([c.smoothness_metrics.smoothness_loss for c in ranked_candidates], dtype=float)

    # Drop any candidates whose metrics are NaN/Inf before computing plot bounds
    valid_mask = np.isfinite(dtw_vals) & np.isfinite(smooth_vals)
    if not valid_mask.any():
        return None
    dtw_vals_valid = dtw_vals[valid_mask]
    smooth_vals_valid = smooth_vals[valid_mask]

    # Extract crank lengths from metadata for colouring (may not exist for all strategies)
    lengths = [
        (c.result.candidate.metadata or {}).get('length', None)
        for c in ranked_candidates
    ]
    has_lengths = all(v is not None for v in lengths)

    # ---- Axes limits with generous padding ----
    x_pad = max((dtw_vals_valid.max() - dtw_vals_valid.min()) * 0.20, dtw_vals_valid.max() * 0.05, 1e-6)
    y_pad = max((smooth_vals_valid.max() - smooth_vals_valid.min()) * 0.20, smooth_vals_valid.max() * 0.05, 1e-6)
    x_min, x_max = dtw_vals_valid.min() - x_pad, dtw_vals_valid.max() + x_pad * 1.5
    y_min, y_max = smooth_vals_valid.min() - y_pad, smooth_vals_valid.max() + y_pad * 1.5

    # ---- Background score grid ----
    gx = np.linspace(x_min, x_max, 400)
    gy = np.linspace(y_min, y_max, 400)
    xx, yy = np.meshgrid(gx, gy)

    x_range = dtw_vals_valid.max() - dtw_vals_valid.min() + 1e-10
    y_range = smooth_vals_valid.max() - smooth_vals_valid.min() + 1e-10
    xx_norm = np.clip((xx - dtw_vals_valid.min()) / x_range, 0, 1)
    yy_norm = np.clip((yy - smooth_vals_valid.min()) / y_range, 0, 1)

    w_c = weights.curve_following if weights is not None else 0.7
    w_s = weights.smoothness if weights is not None else 0.3
    score_grid = w_c * xx_norm + w_s * yy_norm

    # Blue (good/low score) → light pink/red (high score)
    bg_cmap = LinearSegmentedColormap.from_list(
        'pareto_bg',
        [(0.00, '#cce8f4'),
         (0.35, '#e8f4fa'),
         (0.65, '#fce8e8'),
         (1.00, '#f0a8a8')],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.imshow(
        score_grid,
        extent=[x_min, x_max, y_min, y_max],
        origin='lower', aspect='auto',
        cmap=bg_cmap, alpha=0.50, zorder=0,
    )

    # ---- Iso-score contour lines (design trend lines) ----
    contour_levels = np.linspace(0.10, 0.90, 8)
    ax.contour(
        gx, gy, score_grid,
        levels=contour_levels,
        colors=['#7f8c8d'],
        linewidths=0.6,
        linestyles=[':'],
        alpha=0.60,
        zorder=1,
    )

    # ---- Region labels ----
    ax.text(
        x_min + x_pad * 0.3, y_min + y_pad * 0.4,
        'Low Error',
        fontsize=9, color='#1a6b9a', alpha=0.80,
        fontstyle='italic', ha='left', va='bottom', zorder=2,
    )
    ax.text(
        x_max - x_pad * 0.1, (y_min + y_max) * 0.50,
        'High Curve Error',
        fontsize=9, color='#922b21', alpha=0.70,
        fontstyle='italic', ha='right', va='center', rotation=-90, zorder=2,
    )
    ax.text(
        (x_min + x_max) * 0.55, y_max - y_pad * 0.25,
        'High Smoothness Loss',
        fontsize=9, color='#922b21', alpha=0.70,
        fontstyle='italic', ha='center', va='top', zorder=2,
    )

    # ---- Pareto front ----
    pareto_mask = _compute_pareto_front(dtw_vals, smooth_vals)
    pareto_dtw = dtw_vals[pareto_mask]
    pareto_smooth = smooth_vals[pareto_mask]

    sort_idx = np.argsort(pareto_dtw)
    pareto_dtw_sorted = pareto_dtw[sort_idx]
    pareto_smooth_sorted = pareto_smooth[sort_idx]

    # Shade the region below/left of the Pareto front (ideal but unachieved)
    fill_x = np.concatenate([[x_min], pareto_dtw_sorted, [pareto_dtw_sorted[-1], x_min]])
    fill_y = np.concatenate([[pareto_smooth_sorted[0]], pareto_smooth_sorted, [y_min, y_min]])
    ax.fill(fill_x, fill_y, color='#5dade2', alpha=0.12, zorder=2)

    # Step-line
    step_x = [pareto_dtw_sorted[0]]
    step_y = [pareto_smooth_sorted[0]]
    for i in range(1, len(pareto_dtw_sorted)):
        step_x.extend([pareto_dtw_sorted[i], pareto_dtw_sorted[i]])
        step_y.extend([pareto_smooth_sorted[i - 1], pareto_smooth_sorted[i]])
    
    # ax.plot(step_x, step_y, color="#453F3F", linewidth=2.0, linestyle='-.',
    #         alpha=0.90, zorder=5, label='Pareto Front')

    ax.plot(pareto_dtw_sorted, pareto_smooth_sorted,
        color="#453F3F", linewidth=2.0, linestyle='-.', alpha=0.90,
        zorder=5, label='Pareto Front')
    


    ax.scatter(pareto_dtw, pareto_smooth, s=110, facecolors='none',
               edgecolors='#453F3F', linewidths=1.8, zorder=6, label='Pareto-Optimal')

    # ---- Scatter all candidates ----
    if has_lengths:
        length_vals = np.array(lengths, dtype=float)
        unique_lengths = sorted(set(length_vals))
        sc = ax.scatter(
            dtw_vals, smooth_vals,
            c=length_vals, cmap='plasma',
            vmin=min(unique_lengths), vmax=max(unique_lengths),
            s=70, alpha=0.85, zorder=4,
            edgecolors='white', linewidths=0.5,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label('Crank Length (normalized)', fontsize=9)
        cbar.set_ticks(unique_lengths)
        cbar.set_ticklabels([f'{l:.2f}' for l in unique_lengths])
    else:
        ax.scatter(
            dtw_vals, smooth_vals,
            s=70, alpha=0.85, color='#2980b9',
            edgecolors='white', linewidths=0.5,
            zorder=4, label='Candidates',
        )

    # ---- Selected / top-ranked candidate ----
    top = ranked_candidates[0]
    ax.scatter(
        top.result.dtw_distance, top.smoothness_metrics.smoothness_loss,
        marker='*', s=275, color='#f4d03f', edgecolors='#2c3e50', linewidths=0.9,
        zorder=7, label='Selected',
    )

    # ---- Axis styling ----
    smooth_label = 'SPARC Loss (−SPARC)' if smoothness_metric == 'sparc' else 'Composite Smoothness Loss'
    ax.set_xlabel('Dyanmic Time Warping (DTW) Loss', fontsize=11)
    ax.set_ylabel(f'{smooth_label}', fontsize=11)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(direction='out', length=4)

    weight_label = ''
    if weights is not None:
        weight_label = (
            f'\n$w_{{\\mathrm{{curve}}}} = {weights.curve_following:.2f}$,  '
            f'$w_{{\\mathrm{{smooth}}}} = {weights.smoothness:.2f}$'
        )
    ax.set_title(
        f'Pareto Front  —  Sample {sample_index}  '
        f'({len(ranked_candidates)} candidates){weight_label}',
        fontsize=11, pad=10,
    )
    ax.legend(fontsize=9, loc='upper right', framealpha=0.1,
              edgecolor="#000000ff", fancybox=True)


    plt.tight_layout()
    out_path = output_dir / f'pareto_front_sample_{sample_index}.png'
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out_path


# =========================================================================
# SMOOTHNESS METRICS PLOTTING (TOP-LEVEL)
# =========================================================================

def extract_smoothness_metrics(coupler_trajectory):
    """Compute speed, acceleration, and jerk magnitude arrays from a trajectory.

    Returns a dict with keys 'speed', 'accel', 'jerk' (each a 1-D np.ndarray).
    """
    trajectory = np.asarray(coupler_trajectory, dtype=np.float64)
    velocity = np.diff(trajectory, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.diff(velocity, axis=0)
    accel = np.linalg.norm(acceleration, axis=1)
    jerk_vec = np.diff(acceleration, axis=0)
    jerk = np.linalg.norm(jerk_vec, axis=1)
    return {'speed': speed, 'accel': accel, 'jerk': jerk}


def plot_smoothness_metrics_over_time(coupler_trajectory, output_path=None):
    """Plot speed, acceleration, and jerk over time for a candidate."""
    metrics = extract_smoothness_metrics(coupler_trajectory)
    speed, accel_magnitude, jerk_magnitude = metrics['speed'], metrics['accel'], metrics['jerk']
    frames = np.arange(len(speed))
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(frames, speed, label='Speed')
    plt.title('Coupler Speed over Time')
    plt.xlabel('Frame')
    plt.ylabel('Speed')
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(frames[:len(accel_magnitude)], accel_magnitude, label='Acceleration', color='orange')
    plt.title('Coupler Acceleration over Time')
    plt.xlabel('Frame')
    plt.ylabel('Acceleration')
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.plot(frames[:len(jerk_magnitude)], jerk_magnitude, label='Jerk', color='red')
    plt.title('Coupler Jerk over Time')
    plt.xlabel('Frame')
    plt.ylabel('Jerk')
    plt.grid()
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return None



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
    optimize_selected=False,
    plot_pareto=True,
    smoothness_metric='composite',
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
    print(f"Smoothness metric: {smoothness_metric}")
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

    ranked_candidates = rank_candidate_results(valid_results, weights, smoothness_metric=smoothness_metric)

    if optimize_selected:
        print('\nRunning L-BFGS on selected candidate...')
        run_lbfgs_optimization(ranked_candidates[0].result, verbose=verbose)

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

    if plot_pareto:
        pareto_path = plot_pareto_front(
            ranked_candidates, sample_index, output_dir,
            weights=weights, smoothness_metric=smoothness_metric,
        )
        if pareto_path is not None:
            report_paths['pareto_front'] = pareto_path

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
    parser.add_argument('--sample-index', type=int, default=362, help='Validation sample index to evaluate.')
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
    parser.add_argument('--no-pareto-plot', action='store_true', help='Skip the Pareto front PNG.')
    parser.add_argument('--optimize-selected', action='store_true', help='Run L-BFGS-B refinement on the selected (top-ranked) mechanism after ranking.')
    parser.add_argument('--plot-smoothness-metrics', action='store_true', help='Plot speed, acceleration, and jerk over time for the selected candidate.')
    parser.add_argument('--smoothness-metric', choices=['composite', 'sparc'], default='composite', help='Smoothness metric used for ranking: composite (speed_cv + norm_jerk)/2, or sparc (Spectral Arc Length, scale-invariant).')

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
        optimize_selected=args.optimize_selected,
        smoothness_metric=args.smoothness_metric,
        verbose=not args.quiet,
    )

    if result is None:
        print('\nNo selection result was produced.')
    else:
        print(f"\nSelected candidate: {result.selected_candidate.result.candidate.label}")


if __name__ == '__main__':
    main()
