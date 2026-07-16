#!/usr/bin/env python3
"""
Calibration Evaluator
----------------------
Re-project the manually-selected 2D/3D correspondences (exported by
manual_calibrate.py) through one or more candidate transforms and report
per-point reprojection errors.

Usage
-----
  python evaluate_calibration.py correspondences.json \\
      transform_A.json transform_B.json ... \\
      [--labels "Method A" "Method B" ...] \\
      [--out-csv  errors.csv] \\
      [--out-plot errors.png] \\
      [--no-display]

Arguments
---------
  correspondences.json
      Output of manual_calibrate.py — contains the 2D/3D point pairs and
      the camera intrinsics that were active when they were collected.

  transform_*.json  (one or more)
      Each file must have either:
        • "translation" [tx,ty,tz] + "quaternion" [qx,qy,qz,qw]   (compact)
        • a 4×4 row-major "matrix" key                              (alternative)
      Intrinsics keys (fx, fy, …) in the transform file are ignored; the
      correspondences file's intrinsics are always used.

Options
-------
  --labels          Human-readable names for each transform (default: filenames)
  --out-csv PATH    Write a CSV of per-point errors (default: errors.csv)
  --out-plot PATH   Save the plot to a file instead of (or as well as) showing it
  --no-display      Skip the interactive window (useful in headless environments)
  --scale N         Pixel-coordinate scale factor applied to cx/cy/fx/fy when
                    reading the intrinsics (default: value stored in correspondences)

Output CSV columns
------------------
  transform    — label for the candidate transform
  pair_number  — which image/cloud pair the point came from
  pair_label   — e.g. "Pair 1"
  point_index  — 1-based index within that pair
  pt2d_u       — clicked image column  (pixels)
  pt2d_v       — clicked image row     (pixels)
  pt3d_x/y/z  — picked LiDAR point    (metres)
  reproj_u     — reprojected column    (pixels)
  reproj_v     — reprojected row       (pixels)
  error_px     — Euclidean reprojection error (pixels)

Plots produced
--------------
  1. Per-pair box plots  — one subplot per image pair, one box per transform
  2. Summary bar chart   — mean ± std error for every transform, all pairs combined
"""

import argparse
import csv
import json
import os
import sys
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print('Warning: matplotlib not found — plots will be skipped.  '
          'Install with:  pip install matplotlib')


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_correspondences(path: str) -> dict:
    with open(path) as f:
        doc = json.load(f)
    required = {'intrinsics', 'pairs'}
    missing = required - doc.keys()
    if missing:
        raise ValueError(f'correspondences file missing keys: {missing}')
    return doc


def load_transform(path: str) -> np.ndarray:
    """
    Return a 4×4 T_cam_lidar matrix from a JSON file.

    Accepted formats
    ----------------
    • {"translation": [tx,ty,tz], "quaternion": [qx,qy,qz,qw]}
    • {"matrix": [[row0], [row1], [row2], [row3]]}         (4×4 row-major)
    """
    with open(path) as f:
        data = json.load(f)

    if 'matrix' in data:
        T = np.array(data['matrix'], dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f'"matrix" in {path} must be 4×4, got {T.shape}')
        return T

    if 'translation' in data and 'quaternion' in data:
        t = np.array(data['translation'], dtype=np.float64)
        R = Rotation.from_quat(data['quaternion']).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = t
        return T

    raise ValueError(
        f'Transform JSON "{path}" must contain either '
        f'"translation"+"quaternion" or "matrix" keys.\n'
        f'  Found: {list(data.keys())}')


# ══════════════════════════════════════════════════════════════════════════════
# Reprojection
# ══════════════════════════════════════════════════════════════════════════════

def build_camera_params(intrinsics: dict):
    """
    Return (K, dist) from the intrinsics dict stored in correspondences.json.
    The stored values are already at the correct display scale (same scale
    that was active when the points were clicked).
    """
    fx   = float(intrinsics['fx'])
    fy   = float(intrinsics['fy'])
    cx   = float(intrinsics['cx'])
    cy   = float(intrinsics['cy'])
    dist = np.array(intrinsics['dist_coeffs'], dtype=np.float64)
    # Apply the scale factor that was active during collection
    s    = float(intrinsics.get('scale', 1.0))
    K    = np.array([[fx * s, 0,      cx * s],
                     [0,      fy * s, cy * s],
                     [0,      0,      1     ]], dtype=np.float64)
    return K, dist


def reproject(pts3d: np.ndarray, T: np.ndarray,
              K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """
    Project (N,3) LiDAR points through T_cam_lidar into image coordinates.
    Returns (N,2) float array of pixel positions.
    """
    R    = T[:3, :3]
    t    = T[:3,  3]
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    tvec    = t.astype(np.float64).reshape(3, 1)
    proj, _ = cv2.projectPoints(
        pts3d.astype(np.float64), rvec, tvec, K, dist)
    return proj.reshape(-1, 2)


def evaluate_transform(transform_label: str, T: np.ndarray,
                       pairs: list, K: np.ndarray,
                       dist: np.ndarray) -> list[dict]:
    """
    Evaluate one transform against all correspondences.
    Returns a list of per-point result dicts.
    """
    rows = []
    for pair in pairs:
        pnum  = pair['number']
        plabel = pair['label']
        corr   = pair['correspondences']
        if not corr:
            continue

        pts3d = np.array([c['pt3d'] for c in corr], dtype=np.float64)
        pts2d = np.array([c['pt2d'] for c in corr], dtype=np.float64)

        proj  = reproject(pts3d, T, K, dist)
        errs  = np.linalg.norm(proj - pts2d, axis=1)

        for i, (c, (pu, pv), err) in enumerate(zip(corr, proj, errs)):
            rows.append({
                'transform':   transform_label,
                'pair_number': pnum,
                'pair_label':  plabel,
                'point_index': i + 1,
                'pt2d_u':      c['pt2d'][0],
                'pt2d_v':      c['pt2d'][1],
                'pt3d_x':      c['pt3d'][0],
                'pt3d_y':      c['pt3d'][1],
                'pt3d_z':      c['pt3d'][2],
                'reproj_u':    float(pu),
                'reproj_v':    float(pv),
                'error_px':    float(err),
            })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# CSV export
# ══════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    'transform', 'pair_number', 'pair_label', 'point_index',
    'pt2d_u', 'pt2d_v', 'pt3d_x', 'pt3d_y', 'pt3d_z',
    'reproj_u', 'reproj_v', 'error_px',
]


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f'CSV saved → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Console summary table
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(all_rows: list[dict], labels: list[str], pairs: list) -> None:
    """Print a compact summary table to stdout."""
    pair_labels = [p['label'] for p in pairs]

    # Column widths
    lw  = max(len(lb) for lb in labels) + 2
    pw  = max(len(pl) for pl in pair_labels) + 2
    col = 10

    def fmt(v):
        return f'{v:>{col}.2f}' if np.isfinite(v) else f'{"—":>{col}}'

    sep = '─' * (lw + pw + col * 3 + 6)

    print(f'\n{"Transform":<{lw}}  {"Pair":<{pw}}  '
          f'{"Mean px":>{col}}  {"Std px":>{col}}  {"Max px":>{col}}')
    print(sep)

    for label in labels:
        label_rows = [r for r in all_rows if r['transform'] == label]
        # Per-pair breakdown
        for pair in pairs:
            pair_rows = [r for r in label_rows
                         if r['pair_number'] == pair['number']]
            if not pair_rows:
                continue
            errs = np.array([r['error_px'] for r in pair_rows])
            print(f'{label:<{lw}}  {pair["label"]:<{pw}}'
                  f'{fmt(errs.mean())}{fmt(errs.std())}{fmt(errs.max())}')
        # Overall
        all_errs = np.array([r['error_px'] for r in label_rows])
        if len(all_errs):
            print(f'{"":>{lw}}  {"(all pairs)":<{pw}}'
                  f'{fmt(all_errs.mean())}{fmt(all_errs.std())}{fmt(all_errs.max())}')
        print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

# Colour cycle — distinct enough for up to ~8 transforms
_PALETTE = [
    '#4C8EFF', '#FF6B35', '#44BBA4', '#E94F37',
    '#9B5DE5', '#F15BB5', '#FEE440', '#00BBF9',
]


def make_plots(all_rows: list[dict], labels: list[str],
               pairs: list, save_path: str | None,
               no_display: bool,
               pairs_per_page: int = 6) -> None:
    """
    Three outputs:
      Figure set 1 — Per-pair box plots, paginated (pairs_per_page per figure)
      Figure     2 — Overall summary bar chart (mean ± std per transform)
      Figure set 3 — Per-pair CDF of reprojection error (one curve per transform)
    """
    if not HAS_MPL:
        return

    matplotlib.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
    })

    colors   = [_PALETTE[i % len(_PALETTE)] for i in range(len(labels))]
    n_pairs  = len(pairs)
    ncols    = min(pairs_per_page, 3)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def pair_errors(label, pair_number):
        return [r['error_px'] for r in all_rows
                if r['transform'] == label
                and r['pair_number'] == pair_number]

    def save_and_close(fig, suffix):
        if save_path:
            stem, ext = os.path.splitext(save_path)
            p = stem + suffix + (ext or '.png')
            fig.savefig(p, dpi=150, bbox_inches='tight')
            print(f'Plot saved → {p}')
        if not no_display:
            plt.show()
        plt.close(fig)

    # ── Figure set 1: per-pair box plots (paginated) ──────────────────────────
    pages       = (n_pairs + pairs_per_page - 1) // pairs_per_page
    pair_chunks = [pairs[i*pairs_per_page:(i+1)*pairs_per_page]
                   for i in range(pages)]

    for page_idx, chunk in enumerate(pair_chunks):
        n_in_page = len(chunk)
        nrows     = (n_in_page + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 3.5 * nrows),
                                 squeeze=False)
        title = 'Reprojection Error by Pair — Box Plots'
        if pages > 1:
            title += f'  (page {page_idx+1}/{pages})'
        fig.suptitle(title, fontweight='bold', y=1.01)

        for pi, pair in enumerate(chunk):
            ax   = axes[pi // ncols][pi % ncols]
            data = []
            for label in labels:
                errs = pair_errors(label, pair['number'])
                data.append(errs if errs else [float('nan')])

            bp = ax.boxplot(data,
                            patch_artist=True,
                            medianprops=dict(color='white', linewidth=1.5),
                            whiskerprops=dict(linewidth=1),
                            capprops=dict(linewidth=1),
                            flierprops=dict(marker='o', markersize=3, alpha=0.6))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)

            ax.set_title(pair['label'])
            ax.set_ylabel('Error (px)')
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
            ax.yaxis.grid(True, linestyle='--', alpha=0.5)
            ax.set_axisbelow(True)

        for pi in range(n_in_page, nrows * ncols):
            axes[pi // ncols][pi % ncols].set_visible(False)

        fig.tight_layout()
        suffix = f'_boxplot_p{page_idx+1}' if pages > 1 else '_boxplot'
        save_and_close(fig, suffix)

    # ── Figure 2: overall summary bar chart ───────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(max(5, 1.4 * len(labels)), 4))
    fig2.suptitle('Overall Reprojection Error (all pairs)', fontweight='bold')

    means, stds, n_pts = [], [], []
    for label in labels:
        errs = np.array([r['error_px'] for r in all_rows
                         if r['transform'] == label])
        means.append(errs.mean() if len(errs) else float('nan'))
        stds.append(errs.std()   if len(errs) else float('nan'))
        n_pts.append(len(errs))

    x    = np.arange(len(labels))
    bars = ax2.bar(x, means, yerr=stds, capsize=5,
                   color=colors, alpha=0.82, edgecolor='white', linewidth=0.6,
                   error_kw=dict(elinewidth=1.2, ecolor='#444'))

    for bar, m, s, n in zip(bars, means, stds, n_pts):
        if np.isfinite(m):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     m + s + 0.5,
                     f'{m:.2f} px\n(n={n})',
                     ha='center', va='bottom', fontsize=7.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha='right')
    ax2.set_ylabel('Mean reprojection error (px)')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax2.set_axisbelow(True)
    fig2.tight_layout()
    save_and_close(fig2, '_summary')

    # ── Figure set 3: per-pair CDF (paginated) ────────────────────────────────
    # CDF shows "fraction of points with error ≤ X px" — readable at any scale.
    for page_idx, chunk in enumerate(pair_chunks):
        n_in_page = len(chunk)
        nrows     = (n_in_page + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 3.0 * nrows),
                                 squeeze=False)
        title = 'Reprojection Error CDF by Pair'
        if pages > 1:
            title += f'  (page {page_idx+1}/{pages})'
        fig.suptitle(title, fontweight='bold', y=1.01)

        for pi, pair in enumerate(chunk):
            ax = axes[pi // ncols][pi % ncols]
            for label, color in zip(labels, colors):
                errs = np.array(pair_errors(label, pair['number']))
                if len(errs) == 0:
                    continue
                sorted_errs = np.sort(errs)
                cdf         = np.arange(1, len(sorted_errs) + 1) / len(sorted_errs)
                ax.step(sorted_errs, cdf, where='post',
                        color=color, linewidth=1.5, label=label)
                # Mark median
                median = np.median(sorted_errs)
                ax.axvline(median, color=color, linewidth=0.7,
                           linestyle='--', alpha=0.5)

            ax.set_title(pair['label'])
            ax.set_xlabel('Error (px)')
            ax.set_ylabel('Fraction of points')
            ax.set_ylim(0, 1.05)
            ax.set_xlim(left=0)
            ax.yaxis.grid(True, linestyle='--', alpha=0.4)
            ax.xaxis.grid(True, linestyle='--', alpha=0.4)
            ax.set_axisbelow(True)
            if len(labels) > 1:
                ax.legend(fontsize=7)

        for pi in range(n_in_page, nrows * ncols):
            axes[pi // ncols][pi % ncols].set_visible(False)

        fig.tight_layout()
        suffix = f'_cdf_p{page_idx+1}' if pages > 1 else '_cdf'
        save_and_close(fig, suffix)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate camera-LiDAR calibration transforms via '
                    'reprojection error',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--correspondences', '-c', required=True,
                        metavar='FILE',
                        help='correspondences.json from manual_calibrate.py')
    parser.add_argument('transforms', nargs='+',
                        help='One or more transform JSON files to evaluate')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Human-readable label for each transform '
                             '(defaults to filename stem)')
    parser.add_argument('--out-csv', default='errors.csv',
                        help='Output CSV path (default: errors.csv)')
    parser.add_argument('--out-plot', default=None,
                        help='Save plots to this base path '
                             '(suffixes _boxplot/_summary/_perpoint are appended)')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not open interactive plot windows')
    args = parser.parse_args()

    # ── Validate label count ──────────────────────────────────────────────────
    labels = args.labels
    if labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0]
                  for p in args.transforms]
    elif len(labels) != len(args.transforms):
        parser.error(f'--labels: expected {len(args.transforms)} label(s), '
                     f'got {len(labels)}')

    # ── Load correspondences ──────────────────────────────────────────────────
    print(f'Loading correspondences : {args.correspondences}')
    doc   = load_correspondences(args.correspondences)
    pairs = doc['pairs']
    intr  = doc['intrinsics']

    total_pts = sum(len(p['correspondences']) for p in pairs)
    print(f'  {len(pairs)} pair(s)  ·  {total_pts} total correspondences')

    K, dist = build_camera_params(intr)
    print(f'  Intrinsics  fx={intr["fx"]:.2f}  fy={intr["fy"]:.2f}  '
          f'cx={intr["cx"]:.2f}  cy={intr["cy"]:.2f}  '
          f'scale={intr.get("scale", 1.0)}')

    # ── Load and evaluate each transform ─────────────────────────────────────
    all_rows = []
    for path, label in zip(args.transforms, labels):
        print(f'\nLoading transform [{label}] : {path}')
        try:
            T = load_transform(path)
        except Exception as e:
            print(f'  ERROR: {e}  — skipping')
            continue
        print(f'  T_cam_lidar:\n{np.round(T, 5)}')
        rows = evaluate_transform(label, T, pairs, K, dist)
        all_rows.extend(rows)
        errs = np.array([r['error_px'] for r in rows])
        if len(errs):
            print(f'  → {len(errs)} pts  '
                  f'mean={errs.mean():.3f} px  '
                  f'std={errs.std():.3f} px  '
                  f'max={errs.max():.3f} px')

    if not all_rows:
        print('\nNo results to report — check that transforms loaded correctly.')
        sys.exit(1)

    # ── Console summary ───────────────────────────────────────────────────────
    print_summary(all_rows, labels, pairs)

    # ── CSV ───────────────────────────────────────────────────────────────────
    write_csv(all_rows, args.out_csv)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not HAS_MPL:
        print('\nSkipping plots (matplotlib not available).')
    else:
        make_plots(all_rows, labels, pairs,
                   save_path=args.out_plot,
                   no_display=args.no_display)

    print('\nDone.')


if __name__ == '__main__':
    main()