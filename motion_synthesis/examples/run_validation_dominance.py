"""
Run the Section 5.3 dominance experiment over validation samples.

For each of (up to) 50 validation samples we:
  1. Generate the same affine candidate pool.
  2. Record the selected candidate's DTW and L_smooth for four conditions:
     - Random baseline (one uniformly random valid candidate)
     - Curve preset (w_c=1, w_s=0)
     - Balanced preset (w_c=0.5, w_s=0.5)
     - Smooth preset (w_c=0, w_s=1)

Outputs:
  - CSV of per-sample metrics (optional) and a summary table of
    median DTW and median L_smooth per condition (4 rows × 2 metric columns).
  - Same table printed to stdout and saved to a CSV for the paper.

Usage:
  python motion_synthesis/examples/run_validation_dominance.py [--num-samples 50] [--seed 42]
  python motion_synthesis/examples/run_validation_dominance.py --num-samples 50 --workers 4   # ~4x faster (e.g. ~12 min)
  python motion_synthesis/examples/run_validation_dominance.py --num-samples 50 --max-candidates 28   # smaller pool, ~2x faster
  python motion_synthesis/examples/run_validation_dominance.py --num-samples 50 --workers 4 --max-candidates 28   # combine both
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from motion_synthesis.single_mechanism_with_sampling import (
    AffineSamplingStrategy,
    create_validation_split,
    get_sample,
    load_inference_model,
    load_processed_data,
)
from motion_synthesis.smoothness_weighted_selection import (
    SelectionWeights,
    collect_candidate_results,
    rank_candidate_results,
    set_random_seeds,
)

CONDITIONS = [
    ('random', None),           # baseline: random choice from pool
    ('curve', SelectionWeights(1.0, 0.0)),
    ('balanced', SelectionWeights(0.5, 0.5)),
    ('smooth', SelectionWeights(0.0, 1.0)),
]

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'output_smoothness'
SUMMARY_FILENAME = 'validation_dominance_summary.csv'
PER_SAMPLE_FILENAME = 'validation_dominance_per_sample.csv'

# Globals for multiprocessing workers (set by _worker_init)
_worker_data = None
_worker_val_indices = None
_worker_inference = None
_worker_strategy = None
_worker_seed = None
_worker_smoothness_metric = None
_worker_max_candidates = None


def _worker_init(processed_data_path: str | None, seed: int, smoothness_metric: str, max_candidates: int | None):
    """Load model and data once per worker process."""
    global _worker_data, _worker_val_indices, _worker_inference, _worker_strategy
    global _worker_seed, _worker_smoothness_metric, _worker_max_candidates
    set_random_seeds(seed)
    _worker_data = load_processed_data(processed_data_path)
    _, _worker_val_indices = create_validation_split(_worker_data)
    _worker_inference = load_inference_model()
    _worker_strategy = AffineSamplingStrategy()
    _worker_seed = seed
    _worker_smoothness_metric = smoothness_metric
    _worker_max_candidates = max_candidates


def _run_one_sample_worker(sample_index: int) -> tuple[int, dict[str, tuple[float, float]] | None]:
    """Worker entry: run one sample using globals; return (sample_index, result)."""
    result = run_one_sample(
        _worker_inference,
        _worker_data,
        _worker_val_indices,
        sample_index,
        _worker_strategy,
        _worker_smoothness_metric,
        _worker_seed,
        max_candidates=_worker_max_candidates,
    )
    return (sample_index, result)


def run_one_sample(inference, data, val_indices, sample_index: int, strategy,
                   smoothness_metric: str, random_seed: int,
                   max_candidates: int | None = None) -> dict[str, tuple[float, float]] | None:
    """For one validation sample, return dict condition -> (dtw, L_smooth) for selected candidate."""
    sample = get_sample(data, val_indices, sample_index)
    control_points = np.asarray(sample['control_points'], dtype=np.float32)

    valid_results = collect_candidate_results(
        inference,
        control_points,
        sample,
        strategy,
        temperature=0.001,
        max_candidates=max_candidates,
        verbose=False,
    )
    if not valid_results:
        return None

    rng = np.random.RandomState(random_seed + sample_index)
    out = {}

    # Rank once so we have smoothness for every candidate; then random baseline = random from ranked list
    ranked_all = rank_candidate_results(
        valid_results,
        SelectionWeights(0.5, 0.5),
        smoothness_metric=smoothness_metric,
    )
    random_rc = ranked_all[rng.randint(0, len(ranked_all))]
    out['random'] = (random_rc.result.dtw_distance, random_rc.smoothness_metrics.smoothness_loss)
    out['balanced'] = (ranked_all[0].result.dtw_distance, ranked_all[0].smoothness_metrics.smoothness_loss)

    for name, weights in [CONDITIONS[1], CONDITIONS[3]]:  # curve, smooth
        ranked = rank_candidate_results(valid_results, weights, smoothness_metric=smoothness_metric)
        top = ranked[0]
        out[name] = (top.result.dtw_distance, top.smoothness_metrics.smoothness_loss)

    return out


def main():
    parser = argparse.ArgumentParser(description='Run dominance experiment over validation samples.')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of validation samples (default 50).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help='Directory for output CSVs.')
    parser.add_argument('--processed-data-path', type=str, default=None,
                        help='Path to processed_data.pkl (default from project).')
    parser.add_argument('--smoothness-metric', choices=['composite', 'sparc'], default='composite')
    parser.add_argument('--save-per-sample', action='store_true',
                        help='Also save per-sample CSV for debugging.')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel processes (default 1). Use 4 for ~4x speedup.')
    parser.add_argument('--max-candidates', type=int, default=None,
                        help='Cap candidates per sample (default: all ~56). e.g. 28 for ~2x speed, slightly smaller pool.')
    args = parser.parse_args()

    set_random_seeds(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_processed_data(args.processed_data_path)
    _, val_indices = create_validation_split(data)
    n_val = len(val_indices)
    n_run = min(args.num_samples, n_val)
    print(f'Validation set size: {n_val}. Running over {n_run} samples.')
    if args.workers > 1:
        print(f'Using {args.workers} workers.')
    if args.max_candidates is not None:
        print(f'Max candidates per sample: {args.max_candidates}')

    # Collect per-sample results: list of dicts condition -> (dtw, lsmooth)
    rows = []
    if args.workers <= 1:
        print('Loading data and model...')
        inference = load_inference_model()
        strategy = AffineSamplingStrategy()
        for i in range(n_run):
            result = run_one_sample(
                inference, data, val_indices, i, strategy,
                args.smoothness_metric, args.seed,
                max_candidates=args.max_candidates,
            )
            if result is None:
                print(f'  Sample {i}: no valid candidates, skipping.')
                continue
            rows.append((i, result))
            if (i + 1) % 10 == 0:
                print(f'  Completed {i + 1}/{n_run} samples.')
    else:
        ctx = mp.get_context('spawn')
        with ctx.Pool(
            args.workers,
            initializer=_worker_init,
            initargs=(args.processed_data_path, args.seed, args.smoothness_metric, args.max_candidates),
        ) as pool:
            out = pool.map(_run_one_sample_worker, range(n_run), chunksize=1)
        for i, result in out:
            if result is None:
                print(f'  Sample {i}: no valid candidates, skipped.')
                continue
            rows.append((i, result))
        print(f'  Completed {len(rows)}/{n_run} samples with valid results.')

    # rows are (sample_index, result_dict)
    result_dicts = [r[1] for r in rows]

    if not rows:
        print('No samples produced valid results. Exiting.')
        return

    condition_names = [c[0] for c in CONDITIONS]
    # Medians: condition -> (median_dtw, median_lsmooth)
    summary = {}
    for name in condition_names:
        dtws = [r[name][0] for r in result_dicts]
        lsmooths = [r[name][1] for r in result_dicts]
        summary[name] = (float(np.median(dtws)), float(np.median(lsmooths)))

    # Print table
    print('\n' + '=' * 60)
    print('MEDIAN DTW AND MEDIAN L_smooth (composite) BY CONDITION')
    print(f'N = {len(result_dicts)} validation samples')
    print('=' * 60)
    print(f'{"Condition":<14}  {"median DTW":>12}  {"median L_smooth":>16}')
    print('-' * 60)
    for name in condition_names:
        md, ml = summary[name]
        print(f'{name:<14}  {md:>12.4f}  {ml:>16.4f}')
    print('=' * 60)

    # Dominance interpretation
    med_dtw_random, med_lsmooth_random = summary['random']
    all_dominate = True
    for name in condition_names[1:]:
        med_dtw, med_lsmooth = summary[name]
        if med_dtw >= med_dtw_random or med_lsmooth >= med_lsmooth_random:
            all_dominate = False
            break
    if all_dominate:
        print('\nDominance: Every selection condition (curve, balanced, smooth) improves on the')
        print('random baseline on both median DTW and median L_smooth across validation samples.')
    else:
        print('\nDominance: Not all conditions dominate the random baseline on both metrics.')

    # Save summary CSV (4 rows, 2 metric columns)
    summary_path = output_dir / SUMMARY_FILENAME
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'median_DTW', 'median_L_smooth'])
        for name in condition_names:
            w.writerow([name, summary[name][0], summary[name][1]])
    print(f'\nSummary table saved to: {summary_path}')

    if args.save_per_sample:
        per_sample_path = output_dir / PER_SAMPLE_FILENAME
        with open(per_sample_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['sample_index'] + [f'{c}_dtw' for c in condition_names] + [f'{c}_L_smooth' for c in condition_names])
            for sample_idx, r in rows:
                row = [sample_idx]
                for c in condition_names:
                    row.append(r[c][0])
                for c in condition_names:
                    row.append(r[c][1])
                w.writerow(row)
        print(f'Per-sample CSV saved to: {per_sample_path}')


if __name__ == '__main__':
    main()
