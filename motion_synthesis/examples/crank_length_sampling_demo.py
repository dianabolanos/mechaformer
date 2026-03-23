from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_synthesis.single_mechanism_with_sampling import (
    CrankLengthSamplingStrategy,
    build_circle_angles,
)
from motion_synthesis.smoothness_weighted_selection import run_smoothness_selection_demo


def main():
    output_dir = Path(__file__).resolve().parent / 'output_crank_length'
    strategy = CrankLengthSamplingStrategy(
        lengths=[0.5, 0.75, 1.0],
        crank_angles=build_circle_angles(90.0),
    )
    run_smoothness_selection_demo(
        sample_index=0,
        output_dir=output_dir,
        strategy=strategy,
        optimize_for='curve',
        top_report_rows=12,
        visualize_selected=True,
        verbose=True,
    )


if __name__ == '__main__':
    main()
