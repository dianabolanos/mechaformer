"""plot_smoothness_comparison.py

Compare smoothness metrics across selection-weight presets.

Layout
------
Top row   : three trajectory panels (a, b, c) with velocity-arrow glyphs.
            Arrow direction = instantaneous velocity (unit vector).
            Arrow color     = s_t/s̄  (plasma, shared scale across all panels).
            Arrow spacing   = uniform time intervals, so tight = slow, wide = fast.
Bottom row: speed profile s_t/s̄ over normalized simulation progress —
            (b) and (c) on left axis, (a) on secondary right axis.

Expected NPZ files (produced by smoothness_weighted_demo.py --plot-smoothness-metrics):
  smoothness_metrics_sample_<idx>_c1.0_s0.0.npz
  smoothness_metrics_sample_<idx>_c0.5_s0.5.npz
  smoothness_metrics_sample_<idx>_c0.0_s1.0.npz

Usage:
  python motion_synthesis/examples/plot_smoothness_comparison.py --sample-index 596
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'output_smoothness'

ALL_PRESETS = [
    {'curve': 1.0, 'smooth': 0.0, 'label': 'a', 'color': '#e74c3c'},
    {'curve': 0.5, 'smooth': 0.5, 'label': 'b', 'color': '#f39c12'},
    {'curve': 0.0, 'smooth': 1.0, 'label': 'c', 'color': '#2980b9'},
]



def npz_path(output_dir: Path, sample_index: int, curve: float, smooth: float) -> Path:
    return output_dir / f"smoothness_metrics_sample_{sample_index}_c{curve}_s{smooth}.npz"


def load_preset(output_dir: Path, sample_index: int, preset: dict) -> dict | None:
    path = npz_path(output_dir, sample_index, preset['curve'], preset['smooth'])
    if not path.exists():
        print(f"  [missing] {path}")
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _draw_trajectory_panel(ax, traj: np.ndarray, speed: np.ndarray,
                            preset: dict, shared_norm: mcolors.Normalize,
                            n_arrows: int = 22) -> None:
    """Draw coupler trajectory with velocity-arrow glyphs.

    Arrows sampled at equal time intervals — tight spacing = slow, wide = fast.
    Arrow direction = unit velocity vector.
    Arrow color     = s_t/s̄  (shared plasma scale across all panels).
    """
    traj  = np.asarray(traj,  dtype=float)
    speed = np.asarray(speed, dtype=float)

    N   = len(traj)
    vel = np.diff(traj, axis=0)           # (N-1, 2)
    spd = np.linalg.norm(vel, axis=1)     # (N-1,)

    # ── path (gray background) ───────────────────────────────────────────────
    ax.plot(traj[:, 0], traj[:, 1], color='#cccccc', linewidth=0.9, zorder=1)

    # start marker
    ax.plot(traj[0, 0], traj[0, 1], 'o',
            color=preset['color'], markersize=5, zorder=4)

    # ── velocity arrow glyphs ────────────────────────────────────────────────
    stride = max(1, (N - 1) // n_arrows)
    idx = np.arange(0, N - 1, stride)

    x = traj[idx, 0]
    y = traj[idx, 1]
    u = vel[idx, 0]
    v = vel[idx, 1]
    s = spd[idx]

    # Unit direction vectors
    mag   = np.hypot(u, v) + 1e-12
    u_hat = u / mag
    v_hat = v / mag

    # Fixed arrow length = 4 % of bounding-box diagonal
    diag      = np.hypot(np.ptp(traj[:, 0]), np.ptp(traj[:, 1]))
    arrow_len = 0.04 * diag

    # Normalize by own mean for coloring
    mean_spd = spd.mean() if spd.mean() > 0 else 1.0
    s_norm   = s / mean_spd

    ax.quiver(
        x, y,
        u_hat * arrow_len, v_hat * arrow_len,
        s_norm,
        cmap=plt.cm.plasma, norm=shared_norm,
        angles='xy', scale_units='xy', scale=1,
        width=0.02, headwidth=4, headlength=5,
        zorder=3,
    )

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(
        f'({preset["label"]})  $w_c={preset["curve"]},\\ w_s={preset["smooth"]}$',
        fontsize=9, fontweight='bold', color='black', pad=4,
    )
    ax.tick_params(labelsize=7, length=3)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(True, alpha=0.18, linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)


def plot_comparison(sample_index: int, output_dir: Path,
                    save_path: Path | None = None):
    datasets = []
    for preset in ALL_PRESETS:
        data = load_preset(output_dir, sample_index, preset)
        if data is None:
            print(
                f"  Run smoothness_weighted_demo.py with "
                f"--curve-weight {preset['curve']} --smoothness-weight {preset['smooth']} "
                f"--plot-smoothness-metrics first."
            )
        datasets.append(data)

    if all(d is None for d in datasets):
        print("No NPZ files found. Nothing to plot.")
        return None

    # ── Shared colormap norm ─────────────────────────────────────────────────
    # Anchored to (b) and (c) so their variation is fully visible.
    # Mechanism (a) saturates above this range (noted in colorbar label).
    bc_vmax = 2.0
    for data in datasets[1:]:
        if data is not None and 'speed' in data:
            spd      = np.asarray(data['speed'])
            mean_spd = spd.mean()
            if mean_spd > 0:
                bc_vmax = max(bc_vmax, (spd / mean_spd).max())
    vmax        = np.ceil(bc_vmax * 2) / 2   # round up to nearest 0.5
    shared_norm = mcolors.Normalize(vmin=0.0, vmax=vmax)

    # ── Shared x/y limits for top row (union of all trajectories + margin) ──
    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf
    for data in datasets:
        if data is not None and 'trajectory' in data:
            t = np.asarray(data['trajectory'], dtype=float)
            x_min = min(x_min, t[:, 0].min())
            x_max = max(x_max, t[:, 0].max())
            y_min = min(y_min, t[:, 1].min())
            y_max = max(y_max, t[:, 1].max())
    if x_min == np.inf:
        x_min, x_max, y_min, y_max = -1.0, 1.0, -1.0, 1.0
    else:
        dx = max(x_max - x_min, 1e-6)
        dy = max(y_max - y_min, 1e-6)
        margin = 0.06
        x_min -= dx * margin
        x_max += dx * margin
        y_min -= dy * margin
        y_max += dy * margin
        # Equal aspect: use same span so panels are directly comparable
        span = max(x_max - x_min, y_max - y_min)
        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2
        x_min, x_max = x_c - span / 2, x_c + span / 2
        y_min, y_max = y_c - span / 2, y_c + span / 2
    traj_xlim = (x_min, x_max)
    traj_ylim = (y_min, y_max)

    # ── Figure layout: 3 cols × 3 rows (traj | colorbar | speed) ────────────
    from matplotlib.collections import LineCollection

    fig = plt.figure(figsize=(10, 6.0))
    gs  = gridspec.GridSpec(
        3, 3,
        figure=fig,
        height_ratios=[1.9, 0.06, 1.0],
        hspace=0.28, wspace=0.28,
    )

    # ── Row 0: trajectory panels (shared x/y limits) ────────────────────────
    traj_axes = []
    for col, (preset, data) in enumerate(zip(ALL_PRESETS, datasets)):
        ax = fig.add_subplot(gs[0, col])
        traj_axes.append(ax)
        if data is None or 'trajectory' not in data or 'speed' not in data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'({preset["label"]})', fontsize=9)
            continue
        _draw_trajectory_panel(ax, data['trajectory'], data['speed'],
                               preset, shared_norm)
        ax.set_xlim(traj_xlim)
        ax.set_ylim(traj_ylim)

    # ── Row 1: shared colorbar ────────────────────────────────────────────────
    cbar_ax = fig.add_subplot(gs[1, :])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=shared_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    # cbar.set_label(r'$s_t\,/\,\bar{s}$  (scale anchored to b\,\&\,c; a saturates)', fontsize=7.5)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.ax.axvline(1.0, color='white', linewidth=1.4, linestyle='--')

    # ── Row 2: per-column speed profiles ─────────────────────────────────────
    # (b) and (c) share a y-axis limit; (a) auto-scales with a colored axis
    # to signal its scale is fundamentally different.
    bc_ymax = 0.0
    for data in datasets[1:]:
        if data is not None and 'speed' in data:
            spd = np.asarray(data['speed'])
            mean_spd = spd.mean()
            if mean_spd > 0:
                bc_ymax = max(bc_ymax, (spd / mean_spd).max())
    bc_ymax = np.ceil(bc_ymax * 10) / 10 * 1.12

    def _plasma_line(ax, t, spd_norm, lw=2.0):
        pts  = np.stack([t, spd_norm], axis=1).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(segs, cmap=plt.cm.plasma, norm=shared_norm,
                              linewidth=lw, zorder=3)
        lc.set_array(spd_norm)
        ax.add_collection(lc)

    for col, (preset, data) in enumerate(zip(ALL_PRESETS, datasets)):
        ax = fig.add_subplot(gs[2, col])
        if data is None or 'speed' not in data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        spd      = np.asarray(data['speed'])
        mean_spd = spd.mean() if spd.mean() > 0 else 1.0
        spd_norm = spd / mean_spd
        t        = np.linspace(0.0, 1.0, len(spd_norm))

        _plasma_line(ax, t, spd_norm)
        ax.autoscale_view()
        ax.axhline(1.0, color='#555555', linewidth=1.0, linestyle='--', alpha=0.50)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, None if col == 0 else bc_ymax)
        ax.set_xlabel('Fraction of Simulation', fontsize=7.5)
        ax.tick_params(labelsize=7, length=3)
        ax.grid(True, alpha=0.18, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if col == 0:
            ax.set_ylabel(r'$s_t\,/\,\bar{s}$', fontsize=8)
        # elif col == 1:
        #     ax.set_ylabel(r'$s_t\,/\,\bar{s}$', fontsize=8)
        else:
            ax.set_ylabel('')

    # fig.suptitle(
    #     r'Coupler trajectories (top) and speed profiles (bottom)  '
    #     r'$\cdot$  color: $s_t/\bar{s}$  $\cdot$  dashed $=$ mean speed',
    #     fontsize=8.0, y=1.002,
    # )

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"Comparison plot saved to: {save_path}")
        return save_path
    else:
        plt.show()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Compare smoothness metrics for curve-only, balanced, and smooth-only selections.'
    )
    parser.add_argument('--sample-index', type=int, required=True)
    parser.add_argument('--output-dir', type=str, default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    save_path  = None if args.no_save else (
        output_dir / f"smoothness_comparison_sample_{args.sample_index}.png"
    )
    plot_comparison(args.sample_index, output_dir, save_path=save_path)


if __name__ == '__main__':
    main()
