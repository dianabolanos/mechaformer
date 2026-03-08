from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_synthesis.single_mechanism_with_sampling import run_affine_search_demo


def main():
    output_dir = Path(__file__).resolve().parent / 'output_affine'
    run_affine_search_demo(
        sample_index=365,
        output_dir=output_dir,
        run_optimization=False,
        angles=[0.0, 90.0],
        translations=[(0.0, 0.0), (0.3, 0.0)],
        verbose=True,
    )


if __name__ == '__main__':
    main()
