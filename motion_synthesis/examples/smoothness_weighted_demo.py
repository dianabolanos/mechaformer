from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from motion_synthesis.smoothness_weighted_selection import (
    build_arg_parser,
    build_sampling_strategy_from_args,
    extract_smoothness_metrics,
    plot_smoothness_metrics_over_time,
    run_smoothness_selection_demo,
    set_random_seeds,
)

# Default output dir scoped to this examples folder
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'output_smoothness'


def main():
    parser = build_arg_parser()
    # Override the default output dir to keep examples output local
    parser.set_defaults(output_dir=str(_DEFAULT_OUTPUT_DIR), sample_index=362)
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
        plot_pareto=not args.no_pareto_plot,
        smoothness_metric=args.smoothness_metric,
        verbose=not args.quiet,
        #plot_smoothness_metrics=args.plot_smoothness_metrics,
    )

    if result is None:
        print('\nNo selection result was produced.')
    else:
        print(f"\nSelected candidate: {result.selected_candidate.result.candidate.label}")

        # Plot smoothness metrics for selected candidate if requested
        if getattr(args, 'plot_smoothness_metrics', False):
            tag = f"sample_{args.sample_index}_c{args.curve_weight}_s{args.smoothness_weight}"
            trajectory = result.selected_candidate.result.coupler_trajectory

            png_path = Path(args.output_dir) / f"smoothness_metrics_{tag}.png"
            plot_result = plot_smoothness_metrics_over_time(trajectory, output_path=png_path)
            if plot_result is not None:
                print(f"Smoothness metrics plot saved to: {plot_result}")

            metrics = extract_smoothness_metrics(trajectory)
            npz_path = Path(args.output_dir) / f"smoothness_metrics_{tag}.npz"
            np.savez(npz_path, trajectory=np.asarray(trajectory, dtype=np.float64), **metrics)
            print(f"Smoothness metrics data saved to:  {npz_path}")

if __name__ == '__main__':
    main()
