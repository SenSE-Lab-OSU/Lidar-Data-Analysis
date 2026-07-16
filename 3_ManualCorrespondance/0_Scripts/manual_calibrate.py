#!/usr/bin/env python3
"""
Manual LiDAR-Camera Calibration Tool  (multi-pair edition)
--------------------------------------------------------------
Select corresponding points between camera images and LiDAR point clouds,
then solve for the 6-DoF rigid transformation using Algorithm 1 from:
  "General, Single-shot, Target-based, and Targetless Camera-LiDAR Calibration"
  Koide et al., ICRA 2023  https://staff.aist.go.jp/k.koide/assets/pdf/icra2023.pdf

──────────────────────────────────────────────────────────────────────────────
DATA DIRECTORY MODE  (default)
──────────────────────────────────────────────────────────────────────────────
Usage:
  python manual_calibrate.py [--data-dir ./data]

Expected folder layout:
  <data-dir>/camera/cam_1.png   cam_2.png  ...
  <data-dir>/lidar/lidar_1.npy  lidar_2.npy  ...

Files are matched by the trailing number in their names.  Only pairs that
have both a camera image AND a LiDAR file are loaded.

──────────────────────────────────────────────────────────────────────────────
SINGLE-PAIR MODE  (legacy / quick test)
──────────────────────────────────────────────────────────────────────────────
Usage:
  python manual_calibrate.py <camera_image> <lidar_npy>

──────────────────────────────────────────────────────────────────────────────
CORRESPONDENCES-ONLY MODE  (re-solve without reselecting points)
──────────────────────────────────────────────────────────────────────────────
Usage:
  python manual_calibrate.py --correspondences correspondences_full.json \
      [--solve-c] [--solve-delta]

Solves directly against a previously saved correspondences JSON (as written
by this script). No images, LiDAR files, or UI are opened.

──────────────────────────────────────────────────────────────────────────────
OPTIONAL SCALE PARAMETERS  (any mode)
──────────────────────────────────────────────────────────────────────────────
  --solve-c       Also fit a focal-length scale factor c (fx,fy *= c),
                  bounded to +/-10%.
  --solve-delta   Also fit a LiDAR radial-scale factor delta (pts3d *= delta),
                  bounded to +/-5%.
Both default to off, reproducing the original 6-DoF-only solve exactly.
calibration_result.json always includes "c" and "delta" keys (1.0 if unused).

──────────────────────────────────────────────────────────────────────────────
Camera window controls:
  Left-click       Add 2D point  (for the currently displayed pair)
  Right-click      Remove last 2D point

LiDAR window controls:
  Left-click       Pick nearest 3D point  (for the currently displayed pair)
  Right-click      Remove last 3D point
  Scroll wheel     Zoom in/out
  Middle-drag      Pan
  T                Top view   (X-Y plane, Z up)
  F                Front view (X-Z plane, Y into scene)
  S                Side view  (Y-Z plane, X into scene)
  G                Toggle free-rotate mode
  Arrow keys       ← / →  switch pair  (when NOT in free-rotate mode)
                   all four  rotate 5°  (when in free-rotate mode)
  [ / ]            Switch to previous / next pair  (always)
  1–9              Jump directly to pair N

Both windows:
  ENTER            Run solver using ALL pairs combined
  R                Reset current pair's points
  V                Verify prior reprojection for current pair
  Q / ESC          Quit
"""

import sys
import os
import re
import glob
import json
import argparse
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# ── Camera intrinsics ──────────────────────────────────────────────────────────
scale = 1  # Default intrinsics are 1920x1080, so use 2 for 3840x2160
FX   = 1248.8 * scale
FY   = 1244.6 * scale
CX   = 945.08 * scale
CY   = 527.51 * scale
DIST = np.array([0.1949, -0.3245, 0.0, 0.0, 0.0], dtype=np.float64)
K    = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)

# ── Pair colours (BGR for OpenCV) ──────────────────────────────────────────────
COLORS_BGR = [
    (  0, 100, 255),   # orange-red
    ( 50, 205,  50),   # green
    (220,  80,  20),   # blue
    (  0, 200, 255),   # yellow
    (200,  50, 200),   # purple
    (255, 180,   0),   # cyan
    ( 80, 255, 200),   # mint
    (  0, 165, 255),   # orange
    (255,  50, 150),   # pink
    (128, 255,   0),   # lime
]

# ── LiDAR render settings ──────────────────────────────────────────────────────
RENDER_W     = 900
RENDER_H     = 700
POINT_RADIUS = 1        # px radius for each projected point (0 = single pixel)
MAX_POINTS   = 300_000  # downsample above this for render speed

WINDOW_CAM = (
    'Camera  [L-click=add  R-click=remove  ENTER=solve  '
    'R=reset-pair  [/]=switch-pair  Q=quit]'
)
WINDOW_LID = (
    'LiDAR  [L-click=pick  R-click=remove  T/F/S=view  G=free-rotate  '
    '[/]= f/b pair  arrow keys=rotate  scroll=zoom  mid-drag=pan]'
)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_lidar(path: str):
    raw  = np.load(path, allow_pickle=True)
    data = raw.tolist()
    if isinstance(data, list):
        data = data[0]
    cart      = np.array(data.binary.cartesian,    dtype=np.float32)
    intensity = np.array(data.binary.photon_count, dtype=np.float32)
    valid = np.isfinite(cart).all(axis=1) & (np.linalg.norm(cart, axis=1) > 0.01)
    cart      = cart[valid]
    intensity = intensity[valid]
    return cart, intensity


def load_camera(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return img   # keep BGR for OpenCV display


# ══════════════════════════════════════════════════════════════════════════════
# Geometry / solver  (Algorithm 1 -- Koide RANSAC translation)
# ══════════════════════════════════════════════════════════════════════════════

def bearing_vectors(pts2d: np.ndarray) -> np.ndarray:
    pts = pts2d.astype(np.float64).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, DIST).reshape(-1, 2)
    bv  = np.hstack([und, np.ones((len(und), 1))])
    bv /= np.linalg.norm(bv, axis=1, keepdims=True)
    return bv


def rotation_svd(bv: np.ndarray, pts3d: np.ndarray) -> np.ndarray:
    dirs = pts3d / np.linalg.norm(pts3d, axis=1, keepdims=True)
    U, _, Vt = np.linalg.svd(bv.T @ dirs)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R


def ransac_rotation(bv, pts3d, n_iter=1000, thresh_deg=5.0):
    thresh = np.deg2rad(thresh_deg)
    n = len(bv)
    best_R, best_mask = np.eye(3), np.zeros(n, dtype=bool)
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        idx   = rng.choice(n, size=min(2, n), replace=False)
        R_hyp = rotation_svd(bv[idx], pts3d[idx])
        rot   = (R_hyp @ pts3d.T).T
        rot  /= np.linalg.norm(rot, axis=1, keepdims=True)
        dot   = np.clip((bv * rot).sum(axis=1), -1, 1)
        mask  = np.arccos(dot) < thresh
        if mask.sum() > best_mask.sum():
            best_mask, best_R = mask, R_hyp
    if best_mask.sum() >= 2:
        best_R = rotation_svd(bv[best_mask], pts3d[best_mask])
    return best_R, best_mask


def project(pts3d, rvec, tvec, c=1.0, delta=1.0):
    """
    Project 3D points into the image.

    c      : optional focal-length scale factor (models fx,fy being off by
             a uniform scale; principal point cx,cy is left untouched).
    delta  : optional LiDAR radial-scale factor (models r_actual = delta * r
             with theta/phi correct; equivalent to a uniform scale on the
             Cartesian point since x,y,z are all linear in r).
    Both default to 1.0, which reproduces the original behaviour exactly.
    """
    K_eff = K
    if c != 1.0:
        K_eff = K.copy()
        K_eff[0, 0] *= c
        K_eff[1, 1] *= c
    pts = pts3d * delta if delta != 1.0 else pts3d
    p, _ = cv2.projectPoints(pts.astype(np.float64),
                              rvec.astype(np.float64),
                              tvec.astype(np.float64), K_eff, DIST)
    return p.reshape(-1, 2)


def lm_residuals(params, pts3d, pts2d, solve_c=False, solve_delta=False):
    """
    Plain reprojection residuals — no robust weighting during LM.

    params layout: [rvec(3), tvec(3), c?, delta?]  where c and delta are
    only present (in that order) if the corresponding solve_* flag is set.
    """
    rv, tv = params[:3], params[3:6]
    idx = 6
    c = params[idx] if solve_c else 1.0
    if solve_c:
        idx += 1
    delta = params[idx] if solve_delta else 1.0
    proj = project(pts3d, rv, tv, c=c, delta=delta)
    return (proj - pts2d).flatten()


def dlt_solve(pts2d: np.ndarray, pts3d: np.ndarray):
    """
    Direct Linear Transform: build a 2N × 12 system and solve via SVD.
    Returns rvec (3,), tvec (3,) as an initial guess for LM.
    """
    und = cv2.undistortPoints(
        pts2d.astype(np.float64).reshape(-1, 1, 2), K, DIST
    ).reshape(-1, 2)

    N = len(pts3d)
    A = np.zeros((2 * N, 12), dtype=np.float64)
    for i in range(N):
        X, Y, Z = pts3d[i]
        x, y    = und[i]
        A[2*i]     = [ X,  Y,  Z,  1,  0,  0,  0,  0, -x*X, -x*Y, -x*Z, -x]
        A[2*i + 1] = [ 0,  0,  0,  0,  X,  Y,  Z,  1, -y*X, -y*Y, -y*Z, -y]

    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    M, t_h = P[:, :3], P[:, 3]

    U, S, Vt2 = np.linalg.svd(M)
    R_dlt = U @ Vt2
    if np.linalg.det(R_dlt) < 0:
        U[:, -1] *= -1
        R_dlt = U @ Vt2
    t_dlt = t_h / np.mean(S)
    return Rotation.from_matrix(R_dlt).as_rotvec(), t_dlt


# Known approximate transform from a prior calibration run.
# Format: [tx, ty, tz, qx, qy, qz, qw]  (T_cam_lidar, native LiDAR frame)
# Set to None to skip warm-start and rely on DLT only.
#PRIOR_POSE = [0.025329, 0.068218, 0.168532, 0.7060836, 0.00889139, 0.00305466, 0.70806607]
PRIOR_POSE = [0.38861704, -0.03483649, 0.56716535, 0.69748256, -0.02537514, 0.01530879, 0.7159887]


def prior_init():
    """Return rvec, tvec from PRIOR_POSE, or raise if not set."""
    if PRIOR_POSE is None:
        raise ValueError('PRIOR_POSE not set — run a calibration first and update this value')
    t  = np.array(PRIOR_POSE[:3])
    rv = Rotation.from_quat(PRIOR_POSE[3:]).as_rotvec()
    return rv, t


def best_init(pts2d, pts3d, mask):
    """
    Try three initialisation strategies; return the (rvec, tvec) with the
    lowest reprojection error.
    """
    candidates = []

    try:
        rv, t = prior_init()
        err = np.linalg.norm(project(pts3d[mask], rv, t) - pts2d[mask], axis=1).mean()
        candidates.append((err, rv, t, f'prior warm-start ({err:.1f} px)'))
    except Exception as e:
        print(f'  Prior init failed: {e}')

    try:
        rv, t = dlt_solve(pts2d[mask], pts3d[mask])
        err = np.linalg.norm(project(pts3d[mask], rv, t) - pts2d[mask], axis=1).mean()
        if np.isfinite(err):
            candidates.append((err, rv, t, f'DLT ({err:.1f} px)'))
    except Exception as e:
        print(f'  DLT init failed: {e}')

    try:
        bv     = bearing_vectors(pts2d[mask])
        R_ran, _ = ransac_rotation(bv, pts3d[mask])
        rv     = Rotation.from_matrix(R_ran).as_rotvec()
        t      = least_squares(
            lambda t: (project(pts3d[mask], rv, t) - pts2d[mask]).flatten(),
            np.zeros(3), method='lm', max_nfev=500).x
        err = np.linalg.norm(project(pts3d[mask], rv, t) - pts2d[mask], axis=1).mean()
        if np.isfinite(err):
            candidates.append((err, rv, t, f'RANSAC+t ({err:.1f} px)'))
    except Exception as e:
        print(f'  RANSAC init failed: {e}')

    if not candidates:
        raise RuntimeError('All initialisation strategies failed')

    candidates.sort(key=lambda x: x[0])
    best = candidates[0]
    print('  Init strategies tried:')
    for c in candidates:
        marker = ' <-- chosen' if c is best else ''
        print(f'    {c[3]}{marker}')
    return best[1], best[2]


def solve(pts2d: np.ndarray, pts3d: np.ndarray,
          solve_c: bool = False, solve_delta: bool = False) -> dict:
    """
    solve_c     : also fit a focal-length scale factor c (fx,fy *= c).
    solve_delta : also fit a LiDAR radial-scale factor delta (pts3d *= delta).
    Both are off by default, reproducing the original 6-DoF-only behaviour.
    Bounds are kept tight (a few percent) since these are meant to absorb a
    small systematic bias, not act as a free scale — see conversation notes
    on identifiability with t_z / t.
    """
    mask = np.ones(len(pts2d), dtype=bool)
    rv0, t0 = best_init(pts2d, pts3d, mask)

    proj_prior = project(pts3d, rv0, t0)
    prior_errs = np.linalg.norm(proj_prior - pts2d, axis=1)
    print('  Per-point prior reprojection errors (px):')
    for i, e in enumerate(prior_errs):
        flag = '  ← LIKELY BAD' if e > 150 else ''
        print(f'    Point {i+1}: {e:.1f} px{flag}')

    x0     = list(rv0) + list(t0)
    lb     = list(rv0 - 0.5) + list(t0 - 1.0)
    ub     = list(rv0 + 0.5) + list(t0 + 1.0)

    if solve_c:
        x0.append(1.0); lb.append(0.8); ub.append(1.2)
        print('  Fitting optional focal-length scale factor c  (bounds: 0.80–1.20)')
    if solve_delta:
        x0.append(1.0); lb.append(0.95); ub.append(1.05)
        print('  Fitting optional LiDAR radial-scale factor delta  (bounds: 0.95–1.05)')

    x0, lb, ub = np.array(x0), np.array(lb), np.array(ub)

    res = least_squares(lm_residuals, x0,
                        args=(pts3d[mask], pts2d[mask], solve_c, solve_delta),
                        method='trf', bounds=(lb, ub),
                        max_nfev=5000, ftol=1e-10, xtol=1e-10)

    rv, tv = res.x[:3], res.x[3:6]
    idx = 6
    c_opt = res.x[idx] if solve_c else 1.0
    if solve_c:
        idx += 1
    delta_opt = res.x[idx] if solve_delta else 1.0

    if solve_c:
        print(f'  Fitted c     = {c_opt:.5f}')
    if solve_delta:
        print(f'  Fitted delta = {delta_opt:.5f}')

    R_opt  = Rotation.from_rotvec(rv).as_matrix()
    proj   = project(pts3d, rv, tv, c=c_opt, delta=delta_opt)
    errs   = np.linalg.norm(proj - pts2d, axis=1)
    return dict(R=R_opt, t=tv, rvec=rv, tvec=tv, c=c_opt, delta=delta_opt,
                inlier_mask=mask, reproj_errors=errs, projected_pts=proj)


def solve_from_correspondences(corr_path: str,
                               solve_c: bool = False,
                               solve_delta: bool = False,
                               out_path: str = './calibration_result.json') -> dict:
    """
    Run the solver directly against a previously saved correspondences JSON
    (the file written by CalibTool._save_correspondences — e.g.
    'correspondences.json' or a renamed 'correspondences_full.json').
    No images, LiDAR files, or UI are needed; this re-solves against points
    you already picked in an earlier session.
    """
    global FX, FY, CX, CY, DIST, K, scale

    with open(corr_path) as f:
        doc = json.load(f)
    missing = {'intrinsics', 'pairs'} - doc.keys()
    if missing:
        raise ValueError(f'Correspondences file missing keys: {missing}')

    # ── Restore the intrinsics that were active when the points were picked ──
    intr = doc['intrinsics']
    s    = float(intr.get('scale', 1.0))
    FX, FY = float(intr['fx']) * s, float(intr['fy']) * s
    CX, CY = float(intr['cx']) * s, float(intr['cy']) * s
    DIST   = np.array(intr['dist_coeffs'], dtype=np.float64)
    K      = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
    scale  = 1.0   # FX/FY/CX/CY above are already full-pixel-scale

    # ── Flatten all pairs into arrays, same rule as _collect_all_points ──────
    p2, p3, slices, labels = [], [], [], []
    for pair in doc['pairs']:
        corr  = pair['correspondences']
        start = len(p2)
        for c in corr:
            p2.append(c['pt2d'])
            p3.append(c['pt3d'])
        slices.append((start, len(p2)))
        labels.append(pair.get('label', f"Pair {pair.get('number', '?')}"))

    pts2d_arr = np.array(p2, dtype=np.float64)
    pts3d_arr = np.array(p3, dtype=np.float64)
    n_total   = len(pts2d_arr)

    print(f'Loaded {n_total} correspondences from {corr_path} '
          f'across {len(slices)} pair(s)')
    for label, (start, end) in zip(labels, slices):
        print(f'  {label}: {end - start} correspondences')

    if n_total < 3:
        raise RuntimeError(
            f'Need at least 3 total correspondences to solve (have {n_total})')

    print('\n── Running Algorithm 1 (Koide ICRA 2023) ────────────────────')
    result = solve(pts2d_arr, pts3d_arr, solve_c=solve_c, solve_delta=solve_delta)

    errs = result['reproj_errors']
    print('  Per-pair reprojection errors:')
    for label, (start, end) in zip(labels, slices):
        if end > start:
            pe = errs[start:end]
            print(f'    {label}: mean={pe.mean():.2f} px  max={pe.max():.2f} px  '
                  f'({end - start} pts)')
    print(f'  Overall — mean: {errs.mean():.3f} px   max: {errs.max():.3f} px')
    if solve_c:
        print(f'  Fitted focal-length scale c     : {result["c"]:.5f}')
    if solve_delta:
        print(f'  Fitted LiDAR radial-scale delta  : {result["delta"]:.5f}')

    R, t = result['R'], result['t']
    quat = Rotation.from_matrix(R).as_quat()
    out = {
        'translation': t.tolist(),
        'quaternion':  quat.tolist(),
        'fx':          FX / scale,
        'fy':          FY / scale,
        'cx':          CX / scale,
        'cy':          CY / scale,
        'dist_coeffs': DIST.tolist(),
        'c':           float(result['c']),
        'delta':       float(result['delta']),
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    print(f'\nResult saved to {out_path}')
    print(f'T_cam_lidar:\n{np.round(T, 6)}')
    print(f'Human readable rotation (x,y,z): '
          f'{Rotation.from_matrix(R).as_rotvec(degrees=True)}')
    print(f'JSON contents: {json.dumps(out, indent=2)}')
    return out


# ══════════════════════════════════════════════════════════════════════════════
# LiDAR 2-D renderer
# ══════════════════════════════════════════════════════════════════════════════

# Blickfeld LiDAR convention: Y=forward (depth), X=right, Z=up
VIEWS = {
    'T': (0, 1, 2, 'TOP-DOWN    (X right, Y forward)  -- main picking view'),
    'F': (0, 2, 1, 'FRONT-FACE  (X right, Z up)       -- use to verify Z/height'),
    'S': (1, 2, 0, 'SIDE        (Y forward, Z up)      -- use to verify depth'),
}


class LidarRenderer:
    """Renders a point cloud as a 2-D projection with zoom/pan."""

    def __init__(self, pts3d: np.ndarray, intensity: np.ndarray):
        self.pts3d = pts3d
        ni = intensity.astype(np.float32)
        lo, hi = np.percentile(ni, 2), np.percentile(ni, 98)
        ni = np.clip((ni - lo) / max(hi - lo, 1e-6), 0, 1)
        self.intensity_u8 = (ni * 255).astype(np.uint8)

        step = max(1, len(pts3d) // MAX_POINTS)
        self.render_idx = np.arange(0, len(pts3d), step)

        self.view_key    = 'T'
        self.zoom        = 1.0
        self.pan         = np.array([0.0, 0.0])
        self.rot_matrix  = np.eye(3, dtype=np.float64)
        self.free_rotate = False
        self._init_view()

        self.drag_start     = None
        self.pan_start      = None
        self.rot_drag_start = None

    def _init_view(self):
        ax, ay, _ = VIEWS[self.view_key][:3]
        xs = self.pts3d[self.render_idx, ax]
        ys = self.pts3d[self.render_idx, ay]
        cx, cy = xs.mean(), ys.mean()
        span   = max(np.ptp(xs), np.ptp(ys), 1e-3)
        self.zoom = min(RENDER_W, RENDER_H) / span * 0.85
        self.pan  = np.array([cx, cy])

    def set_view(self, key: str):
        self.view_key = key
        self._init_view()

    def world_to_canvas(self, wx, wy):
        px = (wx - self.pan[0]) * self.zoom + RENDER_W / 2
        py = RENDER_H / 2 - (wy - self.pan[1]) * self.zoom
        return px, py

    def canvas_to_world(self, px, py):
        wx = (px - RENDER_W / 2) / self.zoom + self.pan[0]
        wy = (RENDER_H / 2 - py) / self.zoom + self.pan[1]
        return wx, wy

    def _get_projected_pts(self):
        sub = self.pts3d[self.render_idx]
        if self.free_rotate:
            return (self.rot_matrix @ sub.T).T
        return sub

    def render(self, selected_3d: list) -> np.ndarray:
        canvas = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
        ax, ay, _, label = VIEWS[self.view_key]

        sub  = self._get_projected_pts()
        wx   = sub[:, ax]
        wy   = sub[:, ay]
        px, py = self.world_to_canvas(wx, wy)

        in_view = ((px >= 0) & (px < RENDER_W) & (py >= 0) & (py < RENDER_H))
        px  = px[in_view].astype(np.int32)
        py  = py[in_view].astype(np.int32)
        col = self.intensity_u8[self.render_idx][in_view]

        if POINT_RADIUS == 0:
            canvas[py, px] = np.stack([col, col, col], axis=1)
        else:
            for i in range(len(px)):
                c = int(col[i])
                cv2.circle(canvas, (px[i], py[i]), POINT_RADIUS, (c, c, c), -1)

        for i, p3 in enumerate(selected_3d):
            p3r = self.rot_matrix @ p3 if self.free_rotate else p3
            spx, spy = self.world_to_canvas(p3r[ax], p3r[ay])
            color = COLORS_BGR[i % len(COLORS_BGR)]
            cv2.drawMarker(canvas, (int(spx), int(spy)), color,
                           cv2.MARKER_CROSS, 18, 2)
            cv2.circle(canvas, (int(spx), int(spy)), 8, color, 2)
            cv2.putText(canvas, str(i + 1),
                        (int(spx) + 10, int(spy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        mode_str    = 'FREE-ROTATE (arrows=rotate  G=exit)' if self.free_rotate else label
        color_label = (100, 220, 100) if self.free_rotate else (180, 180, 180)
        cv2.putText(canvas, mode_str, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_label, 1, cv2.LINE_AA)
        cv2.putText(canvas, f'pts: {len(selected_3d)}  zoom: {self.zoom:.1f}',
                    (10, RENDER_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
        return canvas

    def pick(self, px: float, py: float):
        ax, ay, _ = VIEWS[self.view_key][:3]
        sub  = self._get_projected_pts()
        spx, spy = self.world_to_canvas(sub[:, ax], sub[:, ay])
        dists = np.hypot(spx - px, spy - py)
        best  = np.argmin(dists)
        if dists[best] < 12:
            return self.pts3d[self.render_idx[best]]
        return None

    def rotate_by(self, dyaw_deg: float, dpitch_deg: float):
        yaw   = np.deg2rad(dyaw_deg)
        pitch = np.deg2rad(dpitch_deg)
        Rz = np.array([[ np.cos(yaw), -np.sin(yaw), 0],
                        [ np.sin(yaw),  np.cos(yaw), 0],
                        [ 0,            0,           1]], dtype=np.float64)
        Rx = np.array([[1, 0,             0            ],
                       [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch),  np.cos(pitch)]], dtype=np.float64)
        self.rot_matrix = Rx @ Rz @ self.rot_matrix
        self._init_view_rotated()

    def _init_view_rotated(self):
        ax, ay, _ = VIEWS[self.view_key][:3]
        rotated = (self.rot_matrix @ self.pts3d[self.render_idx].T).T
        xs, ys  = rotated[:, ax], rotated[:, ay]
        cx, cy  = xs.mean(), ys.mean()
        span    = max(np.ptp(xs), np.ptp(ys), 1e-3)
        self.zoom = min(RENDER_W, RENDER_H) / span * 0.85
        self.pan  = np.array([cx, cy])

    def toggle_free_rotate(self):
        self.free_rotate = not self.free_rotate
        if not self.free_rotate:
            self.rot_matrix = np.eye(3, dtype=np.float64)
        self._init_view()
        print(f"  Free-rotate: {'ON  (all arrows=rotate, G=exit)' if self.free_rotate else 'OFF (←/→=switch pair, T/F/S views restored)'}")

    def zoom_at(self, px: float, py: float, factor: float):
        wx, wy   = self.canvas_to_world(px, py)
        self.zoom = max(0.01, self.zoom * factor)
        new_px, new_py = self.world_to_canvas(wx, wy)
        self.pan[0] += (new_px - px) / self.zoom
        self.pan[1] -= (new_py - py) / self.zoom

    def start_pan(self, px, py):
        self.drag_start = (px, py)
        self.pan_start  = self.pan.copy()

    def update_pan(self, px, py):
        if self.drag_start is None:
            return
        dx = (px - self.drag_start[0]) / self.zoom
        dy = (py - self.drag_start[1]) / self.zoom
        self.pan[0] = self.pan_start[0] - dx
        self.pan[1] = self.pan_start[1] + dy

    def end_pan(self):
        self.drag_start = None
        self.pan_start  = None


# ══════════════════════════════════════════════════════════════════════════════
# Camera overlay renderer
# ══════════════════════════════════════════════════════════════════════════════

def draw_camera(img_bgr, pts2d, result=None,
                pts3d_prior=None, prior_rvec=None, prior_tvec=None,
                pair_label='') -> np.ndarray:
    canvas = img_bgr.copy()

    # ── Pair label banner ────────────────────────────────────────────────────
    if pair_label:
        cv2.putText(canvas, pair_label,
                    (canvas.shape[1] - 220, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 220, 255), 2, cv2.LINE_AA)

    for i, (u, v) in enumerate(pts2d):
        color = COLORS_BGR[i % len(COLORS_BGR)]
        cv2.drawMarker(canvas, (int(u), int(v)), color,
                       cv2.MARKER_CROSS, 18, 2)
        cv2.circle(canvas, (int(u), int(v)), 8, color, 2)
        cv2.putText(canvas, str(i + 1), (int(u) + 10, int(v) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    if result is not None:
        proj = result['projected_pts']
        errs = result['reproj_errors']
        h, w = canvas.shape[:2]
        for i, (pu, pv) in enumerate(proj):
            color = COLORS_BGR[i % len(COLORS_BGR)]
            if not (np.isfinite(pu) and np.isfinite(pv)):
                continue
            pu_i = int(np.clip(pu, -10000, 10000))
            pv_i = int(np.clip(pv, -10000, 10000))
            in_bounds = (0 <= pu_i < w) and (0 <= pv_i < h)
            if in_bounds:
                cv2.drawMarker(canvas, (pu_i, pv_i), color,
                               cv2.MARKER_TILTED_CROSS, 14, 2)
            if i < len(pts2d):
                u0, v0 = pts2d[i]
                pu_c = int(np.clip(pu, 0, w - 1))
                pv_c = int(np.clip(pv, 0, h - 1))
                cv2.line(canvas, (int(u0), int(v0)), (pu_c, pv_c),
                         color, 1, cv2.LINE_AA)
                cv2.putText(canvas, f'{errs[i]:.1f}px',
                            (int(u0) + 10, int(v0) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # ── Prior reprojection overlay ────────────────────────────────────────────
    if pts3d_prior and prior_rvec is not None and prior_tvec is not None:
        try:
            pts3d_arr = np.array(pts3d_prior, dtype=np.float64)
            proj_p    = project(pts3d_arr, prior_rvec, prior_tvec)
            h_img, w_img = canvas.shape[:2]
            for i, (pu, pv) in enumerate(proj_p):
                if not (np.isfinite(pu) and np.isfinite(pv)):
                    continue
                color = COLORS_BGR[i % len(COLORS_BGR)]
                pu_i  = int(np.clip(pu, 0, w_img - 1))
                pv_i  = int(np.clip(pv, 0, h_img - 1))
                in_bounds = (0 <= int(pu) < w_img) and (0 <= int(pv) < h_img)
                cv2.drawMarker(canvas, (pu_i, pv_i), color,
                               cv2.MARKER_DIAMOND, 20, 2)
                label_x = int(np.clip(pu + 12, 0, w_img - 80))
                label_y = int(np.clip(pv - 12, 12, h_img - 4))
                cv2.putText(canvas, f'P{i+1}{"" if in_bounds else " (off)"}',
                            (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        except Exception:
            pass

    n    = len(pts2d)
    hint = '  diamonds=prior-reproject' if (pts3d_prior and prior_rvec is not None) else ''
    cv2.putText(canvas, f'2D pts: {n}  (R-click=remove){hint}',
                (10, canvas.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# Main application
# ══════════════════════════════════════════════════════════════════════════════

class CalibTool:
    """
    Multi-pair calibration tool.

    pairs  — list of dicts, each with keys:
               'img'       : np.ndarray BGR
               'pts3d'     : np.ndarray (N,3)
               'intensity' : np.ndarray (N,)
               'label'     : str  e.g. 'Pair 1'
    """

    def __init__(self, pairs: list, solve_c: bool = False, solve_delta: bool = False):
        if not pairs:
            raise ValueError('No pairs loaded — nothing to do')

        self.pairs       = pairs
        self.n_pairs     = len(pairs)
        self.current     = 0   # index of the displayed pair
        self.solve_c     = solve_c
        self.solve_delta = solve_delta

        # One renderer per pair (pre-built so switching is instant)
        self.renderers = [LidarRenderer(p['pts3d'], p['intensity']) for p in pairs]

        # Per-pair correspondence lists  (properties expose current pair's lists)
        self.pts2d_all: list[list] = [[] for _ in pairs]
        self.pts3d_all: list[list] = [[] for _ in pairs]

        # Solver result for the most recent ENTER press
        self.result      = None
        self.pair_slices = None   # list of (start, end) into result arrays

        cv2.namedWindow(WINDOW_CAM, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WINDOW_LID, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_CAM,
                         min(pairs[0]['img'].shape[1], 1200),
                         min(pairs[0]['img'].shape[0],  800))
        cv2.resizeWindow(WINDOW_LID, RENDER_W, RENDER_H)

        cv2.setMouseCallback(WINDOW_CAM, self._cam_mouse)
        cv2.setMouseCallback(WINDOW_LID, self._lid_mouse)

    # ── Current-pair accessors ─────────────────────────────────────────────────
    # Return the actual list object from the per-pair stores so that in-place
    # operations (.append, .pop, .clear) propagate correctly.

    @property
    def img(self) -> np.ndarray:
        return self.pairs[self.current]['img']

    @property
    def renderer(self) -> LidarRenderer:
        return self.renderers[self.current]

    @property
    def pts2d(self) -> list:
        return self.pts2d_all[self.current]

    @property
    def pts3d_sel(self) -> list:
        return self.pts3d_all[self.current]

    # ── Pair switching ─────────────────────────────────────────────────────────

    def _switch_pair(self, idx: int):
        new_idx = max(0, min(self.n_pairs - 1, idx))
        if new_idx == self.current:
            return
        self.current = new_idx
        # Keep the result visible across pair switches — just the display slice changes
        label = self.pairs[self.current]['label']
        n2    = len(self.pts2d_all[self.current])
        n3    = len(self.pts3d_all[self.current])
        print(f'\n  ── Switched to {label}  '
              f'({n2} 2D pts, {n3} 3D pts)')
        self._refresh()

    # ── Aggregate points across all pairs ─────────────────────────────────────

    def _collect_all_points(self):
        """
        Concatenate all per-pair correspondences.

        Returns
        -------
        pts2d        : (M, 2) float64 — all 2-D clicks
        pts3d        : (M, 3) float64 — all 3-D picks
        pair_slices  : list of (start, end) — index range in the arrays for
                       each pair (slices for pairs with zero correspondences
                       are empty ranges, i.e. start == end)
        """
        p2, p3, slices = [], [], []
        for pts2d, pts3d in zip(self.pts2d_all, self.pts3d_all):
            n     = min(len(pts2d), len(pts3d))
            start = len(p2)
            p2.extend(pts2d[:n])
            p3.extend(pts3d[:n])
            slices.append((start, start + n))
        return (np.array(p2, dtype=np.float64),
                np.array(p3, dtype=np.float64),
                slices)

    def _total_correspondences(self) -> int:
        return sum(min(len(a), len(b))
                   for a, b in zip(self.pts2d_all, self.pts3d_all))

    # ── Pair status bar ────────────────────────────────────────────────────────

    def _draw_pair_bar(self, canvas: np.ndarray):
        """
        Draw a horizontal row of pair boxes on the LiDAR canvas (in-place).
        The active pair is highlighted green.
        """
        n       = self.n_pairs
        # Fit as many boxes as will fill the canvas width
        max_box = max(44, min(90, (RENDER_W - 130) // max(n, 1)))
        box_h   = 22
        y_top   = 34   # sits just below the view-label line

        x = 8
        cv2.putText(canvas, 'Pairs:', (x, y_top + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)
        x += 46

        for i in range(n):
            n_corr = min(len(self.pts2d_all[i]), len(self.pts3d_all[i]))
            is_cur = (i == self.current)

            fill   = (40, 120, 40)  if is_cur else (30, 30, 30)
            border = (200, 255, 200) if is_cur else (80, 80, 80)
            txt_c  = (240, 255, 240) if is_cur else (140, 140, 140)

            bx1, by1 = x, y_top
            bx2, by2 = x + max_box, y_top + box_h

            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), fill, -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), border, 1 if not is_cur else 2)

            # Pair number key hint  + correspondence count
            label = f'[{i+1}] {n_corr}pt'
            cv2.putText(canvas, label, (bx1 + 4, by1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, txt_c, 1, cv2.LINE_AA)

            x += max_box + 3

        # Trailing summary
        total = self._total_correspondences()
        cv2.putText(canvas, f'  total {total} corr  |  [/] ←→ switch',
                    (x + 4, y_top + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _refresh(self):
        rvec_overlay, tvec_overlay = None, None
        if self.result is not None:
            rvec_overlay = self.result['rvec']
            tvec_overlay = self.result['tvec']
        else:
            try:
                rvec_overlay, tvec_overlay = prior_init()
            except ValueError:
                pass

        # Extract the result slice belonging to the current pair
        pair_result = None
        if self.result is not None and self.pair_slices is not None:
            start, end = self.pair_slices[self.current]
            if end > start:
                pair_result = {
                    'projected_pts': self.result['projected_pts'][start:end],
                    'reproj_errors': self.result['reproj_errors'][start:end],
                }

        cam_frame = draw_camera(
            self.img, self.pts2d, pair_result,
            pts3d_prior=self.pts3d_sel if self.pts3d_sel else None,
            prior_rvec=rvec_overlay, prior_tvec=tvec_overlay,
            pair_label=self.pairs[self.current]['label'],
        )

        lid_frame = self.renderer.render(self.pts3d_sel)
        self._draw_pair_bar(lid_frame)

        cv2.imshow(WINDOW_CAM, cam_frame)
        cv2.imshow(WINDOW_LID, lid_frame)

    # ── Mouse callbacks ────────────────────────────────────────────────────────

    def _cam_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts2d.append((float(x), float(y)))
            self.result      = None
            self.pair_slices = None
            self._refresh()
        elif event == cv2.EVENT_RBUTTONDOWN and self.pts2d:
            self.pts2d.pop()
            self.result      = None
            self.pair_slices = None
            self._refresh()

    def _lid_mouse(self, event, x, y, flags, _):
        r = self.renderer
        if event == cv2.EVENT_LBUTTONDOWN:
            pt = r.pick(x, y)
            if pt is not None:
                self.pts3d_sel.append(pt)
                self.result      = None
                self.pair_slices = None
            else:
                print('  No point within click radius — try clicking closer to a dot')
            self._refresh()

        elif event == cv2.EVENT_RBUTTONDOWN and self.pts3d_sel:
            self.pts3d_sel.pop()
            self.result      = None
            self.pair_slices = None
            self._refresh()

        elif event == cv2.EVENT_MBUTTONDOWN:
            r.start_pan(x, y)

        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_MBUTTON):
            r.update_pan(x, y)
            self._refresh()

        elif event == cv2.EVENT_MBUTTONUP:
            r.end_pan()

        elif event == cv2.EVENT_MOUSEWHEEL:
            factor = 1.15 if flags > 0 else 1 / 1.15
            r.zoom_at(x, y, factor)
            self._refresh()

    # ── Prior verifier ─────────────────────────────────────────────────────────

    def _verify_prior(self):
        if not self.pts3d_sel:
            print('  No 3D points selected yet for this pair')
            return
        try:
            rv, t = prior_init()
        except ValueError as e:
            print(f'  {e}')
            return
        pts3d = np.array(self.pts3d_sel, dtype=np.float64)
        proj  = project(pts3d, rv, t)
        h, w  = self.img.shape[:2]
        label = self.pairs[self.current]['label']
        print(f'\n  Prior reprojection of {label} 3D picks onto camera image:')
        for i, (u, v) in enumerate(proj):
            in_frame = (0 <= u < w) and (0 <= v < h)
            flag = '' if in_frame else '  ← OFF SCREEN'
            print(f'    Point {i+1}: ({u:.0f}, {v:.0f}){flag}')
        if self.pts2d:
            print('  Your 2D picks:')
            for i, (u, v) in enumerate(self.pts2d):
                print(f'    Point {i+1}: ({u:.0f}, {v:.0f})')
        print('  (If prior reprojects near your 2D picks, your 3D picks are good)')

    # ── Solver ─────────────────────────────────────────────────────────────────

    def _run_solver(self):
        pts2d_arr, pts3d_arr, slices = self._collect_all_points()
        n_total = len(pts2d_arr)

        # ── Summary of what we're solving with ───────────────────────────────
        print('\n── Multi-pair correspondence summary ─────────────────────────')
        for i, (start, end) in enumerate(slices):
            n       = end - start
            label   = self.pairs[i]['label']
            n2_raw  = len(self.pts2d_all[i])
            n3_raw  = len(self.pts3d_all[i])
            skipped = ''
            if n2_raw != n3_raw:
                skipped = f'  ← WARNING: {n2_raw} 2D vs {n3_raw} 3D, using {n}'
            print(f'  {label}: {n} correspondences{skipped}')
        print(f'  Total: {n_total} correspondences')

        if n_total < 3:
            print(f'  ✗ Need at least 3 total correspondences across all pairs (have {n_total})')
            return

        print('\n── Running Algorithm 1 (Koide ICRA 2023) ────────────────────')
        try:
            self.result = solve(pts2d_arr, pts3d_arr,
                                solve_c=self.solve_c, solve_delta=self.solve_delta)
        except Exception as e:
            print(f'  Solver error: {e}')
            self.result      = None
            self.pair_slices = None
            return

        self.pair_slices = slices

        R, t = self.result['R'], self.result['t']
        errs = self.result['reproj_errors']
        T    = np.eye(4); T[:3, :3] = R; T[:3, 3] = t

        print(f'  RANSAC inliers : {self.result["inlier_mask"].sum()} / {n_total}')

        # Per-pair error breakdown
        print('  Per-pair reprojection errors:')
        for i, (start, end) in enumerate(slices):
            if end > start:
                pe = errs[start:end]
                print(f'    {self.pairs[i]["label"]}: '
                      f'mean={pe.mean():.2f} px  max={pe.max():.2f} px  '
                      f'({end-start} pts)')

        print(f'  Overall — mean: {errs.mean():.3f} px   max: {errs.max():.3f} px')
        if self.solve_c:
            print(f'  Fitted focal-length scale c     : {self.result["c"]:.5f}')
        if self.solve_delta:
            print(f'  Fitted LiDAR radial-scale delta  : {self.result["delta"]:.5f}')
        print(f'\n  4×4 Transform T_cam_lidar (LiDAR → Camera):\n{np.round(T, 6)}')
        print('─────────────────────────────────────────────────────────────\n')
        self._refresh()

    # ── Main event loop ────────────────────────────────────────────────────────

    def run(self):
        self._refresh()
        print(f'\nReady.  {self.n_pairs} pair(s) loaded.')
        print('  Navigation : [ / ]  or  ← / →  to switch pairs  '
              '(arrows rotate instead when free-rotate is ON)')
        print('  Shortcuts  : press 1–9 to jump to a pair directly')
        print('  Solve      : ENTER — uses ALL pairs combined\n')

        while True:
            key = cv2.waitKeyEx(20)

            if key in (13, 10):          # ENTER — solve
                self._run_solver()

            elif key == ord('r'):        # reset current pair
                label = self.pairs[self.current]['label']
                self.pts2d_all[self.current].clear()
                self.pts3d_all[self.current].clear()
                self.result      = None
                self.pair_slices = None
                print(f'  Reset {label}')
                self._refresh()

            elif key in (ord('v'), ord('V')):
                self._verify_prior()

            # ── View controls ───────────────────────────────────────────────
            elif key in (ord('t'), ord('T')):
                self.renderer.set_view('T'); self._refresh()
            elif key in (ord('f'), ord('F')):
                self.renderer.set_view('F'); self._refresh()
            elif key in (ord('s'), ord('S')):
                self.renderer.set_view('S'); self._refresh()
            elif key in (ord('g'), ord('G')):
                self.renderer.toggle_free_rotate(); self._refresh()

            # ── Pair navigation — [ ] always switch ─────────────────────────
            elif key == ord('['):
                self._switch_pair(self.current - 1)
            elif key == ord(']'):
                self._switch_pair(self.current + 1)

            # ── Number keys 1–9 jump to pair directly ───────────────────────
            elif ord('1') <= key <= ord('9'):
                self._switch_pair(key - ord('1'))

            # ── Arrow keys: rotate when free-rotate is ON, switch pair otherwise
            elif key in (65361, 2424832):   # left arrow  (Linux / Windows)
                if self.renderer.free_rotate:
                    self.renderer.rotate_by(dyaw_deg=-5, dpitch_deg=0)
                    self._refresh()
                else:
                    self._switch_pair(self.current - 1)
            elif key in (65363, 2555904):   # right arrow
                if self.renderer.free_rotate:
                    self.renderer.rotate_by(dyaw_deg=+5, dpitch_deg=0)
                    self._refresh()
                else:
                    self._switch_pair(self.current + 1)
            elif key in (65362, 2490368):   # up arrow (only active in free-rotate)
                if self.renderer.free_rotate:
                    self.renderer.rotate_by(dyaw_deg=0, dpitch_deg=-5)
                    self._refresh()
            elif key in (65364, 2621440):   # down arrow
                if self.renderer.free_rotate:
                    self.renderer.rotate_by(dyaw_deg=0, dpitch_deg=+5)
                    self._refresh()

            elif key in (ord('q'), ord('Q'), 27):   # Q / ESC
                break

            # Quit if either window was closed by the user
            if cv2.getWindowProperty(WINDOW_CAM, cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.getWindowProperty(WINDOW_LID, cv2.WND_PROP_VISIBLE) < 1:
                break

        if self.result is not None:
            R    = self.result['R']
            t    = self.result['t']
            quat = Rotation.from_matrix(R).as_quat()   # [qx, qy, qz, qw]

            out = {
                'translation': t.tolist(),
                'quaternion':  quat.tolist(),
                'fx':          FX / scale,
                'fy':          FY / scale,
                'cx':          CX / scale,
                'cy':          CY / scale,
                'dist_coeffs': DIST.tolist(),
                # Optional scale corrections — always present (default 1.0)
                # so downstream tools (evaluate_calibration.py,
                # visualize_calibration.py) can read these keys unconditionally.
                'c':           float(self.result.get('c', 1.0)),
                'delta':       float(self.result.get('delta', 1.0)),
            }

            out_path = './calibration_result.json'
            with open(out_path, 'w') as f:
                json.dump(out, f, indent=2)

            T = np.eye(4)
            T[:3, :3] = R
            T[:3,  3] = t
            print(f'\nResult saved to {out_path}')
            print(f'T_cam_lidar:\n{np.round(T, 6)}')
            print(f'Human readable rotation (x,y,z): '
                  f'{Rotation.from_matrix(R).as_rotvec(degrees=True)}')
            print(f'JSON contents: {json.dumps(out, indent=2)}')

        # ── Always save correspondences (even if no solve was run) ────────────
        self._save_correspondences('./correspondences.json')
        cv2.destroyAllWindows()

    def _save_correspondences(self, out_path: str) -> None:
        """
        Write every per-pair 2D/3D correspondence to a JSON file so that
        evaluate_calibration.py can later test arbitrary transforms against them.

        Schema
        ------
        {
          "intrinsics": { "fx", "fy", "cx", "cy", "dist_coeffs", "scale" },
          "pairs": [
            {
              "number": 1,
              "label":  "Pair 1",
              "correspondences": [
                { "pt2d": [u, v], "pt3d": [x, y, z] },
                ...
              ]
            },
            ...
          ]
        }

        Only pairs with at least one matched correspondence are included.
        If a pair has an unequal number of 2D and 3D points the shorter
        list determines how many correspondences are saved (same rule the
        solver uses).
        """
        pairs_out = []
        for i, pair_meta in enumerate(self.pairs):
            pts2d = self.pts2d_all[i]
            pts3d = self.pts3d_all[i]
            n     = min(len(pts2d), len(pts3d))
            if n == 0:
                continue
            corr = [
                {'pt2d': [float(v) for v in pts2d[j]],
                'pt3d': [float(v) for v in pts3d[j]]}
                for j in range(n)
            ]
            pairs_out.append({
                'number':          pair_meta['number'],
                'label':           pair_meta['label'],
                'correspondences': corr,
            })

        total = sum(len(p['correspondences']) for p in pairs_out)

        doc = {
            'intrinsics': {
                'fx':          FX / scale,
                'fy':          FY / scale,
                'cx':          CX / scale,
                'cy':          CY / scale,
                'dist_coeffs': DIST.tolist(),
                'scale':       scale,
            },
            'pairs': pairs_out,
        }

        with open(out_path, 'w') as f:
            print(f"DEBUG: doc of type {type(doc)}")
            print(f"DEBUG: d of type {type(f)}")
            json.dump(doc, f, indent=2)

        print(f'\nCorrespondences saved to {out_path}  '
              f'({total} pts across {len(pairs_out)} pair(s))')


# ══════════════════════════════════════════════════════════════════════════════
# Pair discovery
# ══════════════════════════════════════════════════════════════════════════════

def discover_pairs(data_dir: str) -> list[dict]:
    """
    Scan  <data_dir>/camera/cam_N.png  and  <data_dir>/lidar/lidar_N.npy
    for all matched N, returning a sorted list of metadata dicts.
    """
    cam_dir   = os.path.join(data_dir, 'camera')
    lidar_dir = os.path.join(data_dir, 'lidar')

    if not os.path.isdir(cam_dir):
        raise FileNotFoundError(
            f'Camera directory not found: {cam_dir}\n'
            f'Expected layout: {data_dir}/camera/cam_N.png')
    if not os.path.isdir(lidar_dir):
        raise FileNotFoundError(
            f'LiDAR directory not found: {lidar_dir}\n'
            f'Expected layout: {data_dir}/lidar/lidar_N.npy')

    def extract_number(path: str) -> int | None:
        m = re.search(r'(\d+)', os.path.splitext(os.path.basename(path))[0])
        return int(m.group(1)) if m else None

    cam_map   = {}
    for p in glob.glob(os.path.join(cam_dir,   'cam_*.png')):
        n = extract_number(p)
        if n is not None:
            cam_map[n] = p

    lidar_map = {}
    for p in glob.glob(os.path.join(lidar_dir, 'lidar_*.npy')):
        n = extract_number(p)
        if n is not None:
            lidar_map[n] = p

    common = sorted(set(cam_map) & set(lidar_map))

    if not common:
        cam_found   = sorted(cam_map)
        lidar_found = sorted(lidar_map)
        raise FileNotFoundError(
            f'No matching cam_N / lidar_N pairs found.\n'
            f'  Camera indices  : {cam_found}\n'
            f'  LiDAR  indices  : {lidar_found}')

    only_cam   = sorted(set(cam_map)   - set(lidar_map))
    only_lidar = sorted(set(lidar_map) - set(cam_map))
    if only_cam:
        print(f'  Warning: camera images with no matching LiDAR: {only_cam}')
    if only_lidar:
        print(f'  Warning: LiDAR files with no matching camera: {only_lidar}')

    return [
        {'cam_path': cam_map[n], 'lidar_path': lidar_map[n],
         'number': n, 'label': f'Pair {n}'}
        for n in common
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Manual LiDAR-Camera calibration — multi-pair edition '
                    '(Koide Algorithm 1, ICRA 2023)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--data-dir', default='./data', metavar='DIR',
        help='Root directory containing camera/ and lidar/ sub-folders '
             '(default: ./data).  Ignored when positional args are given.')
    # Legacy single-pair positional args (still supported)
    parser.add_argument('camera_image', nargs='?', default=None,
                        help='[legacy] Camera image (.png / .jpg)')
    parser.add_argument('lidar_npy',    nargs='?', default=None,
                        help='[legacy] Blickfeld LiDAR .npy file')

    # ── Optional extra scale parameters ────────────────────────────────────────
    parser.add_argument('--solve-c', action='store_true',
                        help='Also fit a focal-length scale factor c '
                             '(fx,fy *= c) during the solve, bounded to +/-10%%.')
    parser.add_argument('--solve-delta', action='store_true',
                        help='Also fit a LiDAR radial-scale factor delta '
                             '(pts3d *= delta) during the solve, bounded to +/-5%%.')

    # ── Solve directly from a saved correspondences file, no UI ───────────────
    parser.add_argument('--correspondences', default=None, metavar='FILE',
                        help='Skip image/LiDAR loading and the picking UI '
                             'entirely; solve directly against a previously '
                             'saved correspondences JSON (e.g. '
                             'correspondences.json or correspondences_full.json).')
    args = parser.parse_args()

    # ── Correspondences-only mode: solve and exit, no UI ──────────────────────
    if args.correspondences:
        solve_from_correspondences(args.correspondences,
                                   solve_c=args.solve_c,
                                   solve_delta=args.solve_delta)
        return

    # ── Decide which mode to run ───────────────────────────────────────────────
    if args.camera_image and args.lidar_npy:
        # Legacy single-pair mode
        pairs_meta = [{
            'cam_path':   args.camera_image,
            'lidar_path': args.lidar_npy,
            'number':     1,
            'label':      'Pair 1',
        }]
        print('Single-pair mode (legacy positional arguments)')
    else:
        # Directory mode
        data_dir = os.path.abspath(args.data_dir)
        print(f'Scanning data directory: {data_dir}')
        pairs_meta = discover_pairs(data_dir)
        print(f'  Found {len(pairs_meta)} matched pair(s)')

    # ── Load all pairs ─────────────────────────────────────────────────────────
    pairs = []
    for meta in pairs_meta:
        label = meta['label']
        print(f'\nLoading {label}:')

        print(f'  Camera : {meta["cam_path"]}')
        img = load_camera(meta['cam_path'])
        print(f'    {img.shape[1]} × {img.shape[0]} px')

        print(f'  LiDAR  : {meta["lidar_path"]}')
        pts3d, intensity = load_lidar(meta['lidar_path'])
        print(f'    {len(pts3d):,} points  (rendering up to {MAX_POINTS:,})')

        pairs.append({
            'img':       img,
            'pts3d':     pts3d,
            'intensity': intensity,
            'label':     label,
            'number':    meta['number'],
        })

    print(f'\nAll {len(pairs)} pair(s) loaded.  Starting UI …\n')
    CalibTool(pairs, solve_c=args.solve_c, solve_delta=args.solve_delta).run()


if __name__ == '__main__':
    main()