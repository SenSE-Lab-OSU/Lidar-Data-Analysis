#!/usr/bin/env python3
"""
Manual LiDAR-Camera Calibration Tool
--------------------------------------
Select corresponding points between a camera image and a LiDAR point cloud,
then solve for the 6-DoF rigid transformation using Algorithm 1 from:
  "General, Single-shot, Target-based, and Targetless Camera-LiDAR Calibration"
  Koide et al., ICRA 2023  https://staff.aist.go.jp/k.koide/assets/pdf/icra2023.pdf

Usage:
  python manual_calibrate.py <camera_image> <lidar_npy>

Camera window controls:
  Left-click       Add 2D point
  Right-click      Remove last 2D point

LiDAR window controls:
  Left-click       Pick nearest 3D point
  Right-click      Remove last 3D point
  Scroll wheel     Zoom in/out
  Middle-drag      Pan
  T                Top view   (X-Y plane, Z up)
  F                Front view (X-Z plane, Y into scene)
  S                Side view  (Y-Z plane, X into scene)
  G                Toggle free-rotate mode (overrides T/F/S)
  Arrow keys       Rotate view 5° per press (in free-rotate mode)

Both windows:
  ENTER            Run solver
  R                Reset all points
  Q / ESC          Quit
"""

import sys
import argparse
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# ── Camera intrinsics ──────────────────────────────────────────────────────────
FX   = 1248.8
FY   = 1244.6
CX   = 945.08
CY   = 527.51
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
RENDER_W    = 900
RENDER_H    = 700
POINT_RADIUS = 1      # px radius for each projected point (0 = single pixel, faster)
MAX_POINTS  = 300_000 # downsample above this for render speed

WINDOW_CAM  = 'Camera Image  [left-click=add  right-click=remove  ENTER=solve  R=reset  Q=quit  | diamonds=prior-reproject-of-3D-picks]'
WINDOW_LID  = 'LiDAR Cloud  [left-click=pick  right-click=remove  T/F/S=view  G=free-rotate  arrows=rotate  V=verify  scroll=zoom  mid-drag=pan]'


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_lidar(path: str):
    raw  = np.load(path, allow_pickle=True)
    data = raw.tolist()
    # Some files are saved as a list, others as a scalar object
    if isinstance(data, list):
        data = data[0]
    cart      = np.array(data.binary.cartesian,    dtype=np.float32)
    intensity = np.array(data.binary.photon_count, dtype=np.float32)
    valid = np.isfinite(cart).all(axis=1) & (np.linalg.norm(cart, axis=1) > 0.01)
    cart  = cart[valid]
    intensity = intensity[valid]
    # Negate X to match camera convention (+X = right).
    # The prior calibration was done with this convention; remove this line
    # if you re-derive the prior from scratch on your own data.
    cart[:, 0] *= -1
    return cart, intensity


def load_camera(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return img   # keep BGR for OpenCV display


# ══════════════════════════════════════════════════════════════════════════════
# Geometry / solver  (Algorithm 1)
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


def project(pts3d, rvec, tvec):
    p, _ = cv2.projectPoints(pts3d.astype(np.float64),
                              rvec.astype(np.float64),
                              tvec.astype(np.float64), K, DIST)
    return p.reshape(-1, 2)


def lm_residuals(params, pts3d, pts2d):
    """Plain reprojection residuals — no robust weighting during LM.
    Cauchy weighting was destabilising convergence from a good warm-start."""
    proj = project(pts3d, params[:3], params[3:])
    return (proj - pts2d).flatten()


def dlt_solve(pts2d: np.ndarray, pts3d: np.ndarray):
    """
    Direct Linear Transform: build a 2N x 12 system and solve via SVD
    to get a full P = [R|t] projection matrix (K already divided out via
    undistortPoints).  Returns rvec (3,), tvec (3,) as a close initial
    guess for LM, avoiding the near-zero-translation trap.
    """
    und = cv2.undistortPoints(
        pts2d.astype(np.float64).reshape(-1, 1, 2), K, DIST
    ).reshape(-1, 2)   # normalised coords x/z, y/z

    N = len(pts3d)
    A = np.zeros((2 * N, 12), dtype=np.float64)
    for i in range(N):
        X, Y, Z = pts3d[i]
        x, y    = und[i]
        A[2*i]     = [ X,  Y,  Z,  1,  0,  0,  0,  0, -x*X, -x*Y, -x*Z, -x]
        A[2*i + 1] = [ 0,  0,  0,  0,  X,  Y,  Z,  1, -y*X, -y*Y, -y*Z, -y]

    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)

    M   = P[:, :3]
    t_h = P[:, 3]

    # Recover proper rotation via SVD and scale
    U, S, Vt2 = np.linalg.svd(M)
    R_dlt = U @ Vt2
    if np.linalg.det(R_dlt) < 0:
        U[:, -1] *= -1
        R_dlt = U @ Vt2
    scale = np.mean(S)
    t_dlt = t_h / scale

    rvec = Rotation.from_matrix(R_dlt).as_rotvec()
    return rvec, t_dlt


# Known approximate transform from prior calibration run.
# Used as warm-start for LM — edit if your setup changes significantly.
# Format: [tx, ty, tz, qx, qy, qz, qw]  (T_lidar_camera)
PRIOR_POSE = [-0.0010442037923634948, -0.21121605091124662, 0.0840489534307288,
              -0.7066547713046278, -0.013130005613761948, 0.006348149443893755, 0.7074081835430127]


def prior_init():
    """Return rvec, tvec from PRIOR_POSE."""
    t  = np.array(PRIOR_POSE[:3])
    rv = Rotation.from_quat(PRIOR_POSE[3:]).as_rotvec()
    return rv, t


def best_init(pts2d, pts3d, mask):
    """
    Try three initialisation strategies in order of reliability.
    Returns (rvec, tvec, label) for whichever gives the lowest reprojection error.
    """
    candidates = []

    # 1. Prior warm-start
    try:
        rv, t = prior_init()
        err = np.linalg.norm(project(pts3d[mask], rv, t) - pts2d[mask], axis=1).mean()
        candidates.append((err, rv, t, f'prior warm-start ({err:.1f} px)'))
    except Exception as e:
        print(f'  Prior init failed: {e}')

    # 2. DLT
    try:
        rv, t = dlt_solve(pts2d[mask], pts3d[mask])
        err = np.linalg.norm(project(pts3d[mask], rv, t) - pts2d[mask], axis=1).mean()
        if np.isfinite(err):
            candidates.append((err, rv, t, f'DLT ({err:.1f} px)'))
    except Exception as e:
        print(f'  DLT init failed: {e}')

    # 3. RANSAC rotation + coarse translation
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
    print(f'  Init strategies tried:')
    for c in candidates:
        marker = ' <-- chosen' if c is best else ''
        print(f'    {c[3]}{marker}')
    return best[1], best[2]


def solve(pts2d: np.ndarray, pts3d: np.ndarray) -> dict:
    mask = np.ones(len(pts2d), dtype=bool)

    rv0, t0 = best_init(pts2d, pts3d, mask)

    # ── Per-point prior errors — helps spot bad correspondences ──────────────
    proj_prior = project(pts3d, rv0, t0)
    prior_errs = np.linalg.norm(proj_prior - pts2d, axis=1)
    print(f'  Per-point prior reprojection errors (px):')
    for i, e in enumerate(prior_errs):
        flag = '  ← LIKELY BAD' if e > 150 else ''
        print(f'    Point {i+1}: {e:.1f} px{flag}')

    # ── Joint 6-DoF LM refinement ─────────────────────────────────────────────
    # Use trf (trust-region) instead of lm so we can bound the solution
    # within a reasonable neighbourhood of the warm-start
    x0     = np.concatenate([rv0, t0])
    margin = np.array([0.5, 0.5, 0.5,    # rotation  ±0.5 rad (~28°)
                       1.0, 1.0, 1.0])   # translation ±1.0 m
    lb, ub = x0 - margin, x0 + margin
    res = least_squares(lm_residuals, x0,
                        args=(pts3d[mask], pts2d[mask]),
                        method='trf', bounds=(lb, ub),
                        max_nfev=5000, ftol=1e-10, xtol=1e-10)
    rv, tv = res.x[:3], res.x[3:]
    R_opt  = Rotation.from_rotvec(rv).as_matrix()
    proj   = project(pts3d, rv, tv)
    errs   = np.linalg.norm(proj - pts2d, axis=1)
    return dict(R=R_opt, t=tv, rvec=rv, tvec=tv,
                inlier_mask=mask, reproj_errors=errs, projected_pts=proj)


# ══════════════════════════════════════════════════════════════════════════════
# LiDAR 2-D renderer
# ══════════════════════════════════════════════════════════════════════════════

# Blickfeld LiDAR convention: Y=forward (depth), X=right, Z=up
# Views are labelled from the camera's perspective looking into the scene
VIEWS = {
    'T': (0, 1, 2, 'TOP-DOWN    (X right, Y forward)  -- main picking view'),
    'F': (0, 2, 1, 'FRONT-FACE  (X right, Z up)       -- use to verify Z/height'),
    'S': (1, 2, 0, 'SIDE        (Y forward, Z up)      -- use to verify depth'),
}


class LidarRenderer:
    """Renders a point cloud as a 2-D projection with zoom/pan."""

    def __init__(self, pts3d: np.ndarray, intensity: np.ndarray):
        self.pts3d = pts3d
        # normalised intensity → greyscale lookup (0-255)
        ni = intensity.astype(np.float32)
        lo, hi = np.percentile(ni, 2), np.percentile(ni, 98)
        ni = np.clip((ni - lo) / max(hi - lo, 1e-6), 0, 1)
        self.intensity_u8 = (ni * 255).astype(np.uint8)

        # downsample for rendering
        step = max(1, len(pts3d) // MAX_POINTS)
        self.render_idx = np.arange(0, len(pts3d), step)

        self.view_key   = 'T'
        self.zoom       = 1.0
        self.pan        = np.array([0.0, 0.0])   # in world units
        # Free-rotation mode: accumulated rotation matrix applied to all points
        # before projection.  Identity = use raw axes (T/F/S views).
        self.rot_matrix  = np.eye(3, dtype=np.float64)
        self.free_rotate = False   # toggled by G key
        self._init_view()

        # interaction state
        self.drag_start   = None
        self.pan_start    = None
        self.rot_drag_start = None   # for right-drag rotation

    def _init_view(self):
        """Set pan/zoom to fit the point cloud in the current view."""
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

    def world_to_canvas(self, wx: np.ndarray, wy: np.ndarray):
        """World X,Y → canvas pixel coords."""
        px = (wx - self.pan[0]) * self.zoom + RENDER_W / 2
        py = RENDER_H / 2 - (wy - self.pan[1]) * self.zoom   # flip Y
        return px, py

    def canvas_to_world(self, px: float, py: float):
        wx = (px - RENDER_W / 2) / self.zoom + self.pan[0]
        wy = (RENDER_H / 2 - py) / self.zoom + self.pan[1]
        return wx, wy

    def _get_projected_pts(self):
        """Return (N,3) points after applying rotation (if in free-rotate mode)."""
        sub = self.pts3d[self.render_idx]
        if self.free_rotate:
            return (self.rot_matrix @ sub.T).T
        return sub

    def render(self, selected_3d: list, hover_pt=None) -> np.ndarray:
        canvas = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
        ax, ay, _, label = VIEWS[self.view_key]

        sub  = self._get_projected_pts()
        wx   = sub[:, ax]
        wy   = sub[:, ay]
        px, py = self.world_to_canvas(wx, wy)

        # clip to canvas
        in_view = ((px >= 0) & (px < RENDER_W) & (py >= 0) & (py < RENDER_H))
        px  = px[in_view].astype(np.int32)
        py  = py[in_view].astype(np.int32)
        col = self.intensity_u8[self.render_idx][in_view]

        # draw each point as a grey dot (intensity-coloured)
        if POINT_RADIUS == 0:
            canvas[py, px] = np.stack([col, col, col], axis=1)
        else:
            for i in range(len(px)):
                c = int(col[i])
                cv2.circle(canvas, (px[i], py[i]), POINT_RADIUS, (c, c, c), -1)

        # draw selected points (apply same rotation as cloud)
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

        # view label
        mode_str = 'FREE-ROTATE (arrows=rotate  G=exit)' if self.free_rotate else label
        color_label = (100, 220, 100) if self.free_rotate else (180, 180, 180)
        cv2.putText(canvas, mode_str, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_label, 1, cv2.LINE_AA)
        cv2.putText(canvas, f'pts: {len(selected_3d)}  zoom: {self.zoom:.1f}',
                    (10, RENDER_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
        return canvas

    def pick(self, px: float, py: float) -> np.ndarray | None:
        """Return the 3D point whose projection is closest to canvas pixel (px,py).
        Always returns the original (unrotated) 3D point for use in the solver."""
        ax, ay, _ = VIEWS[self.view_key][:3]
        sub  = self._get_projected_pts()   # rotated for display matching
        spx, spy = self.world_to_canvas(sub[:, ax], sub[:, ay])
        dists = np.hypot(spx - px, spy - py)
        best  = np.argmin(dists)
        if dists[best] < 12:
            return self.pts3d[self.render_idx[best]]   # original unrotated point
        return None

    def rotate_by(self, dyaw_deg: float, dpitch_deg: float):
        """Accumulate a yaw (left/right) and pitch (up/down) rotation."""
        yaw   = np.deg2rad(dyaw_deg)
        pitch = np.deg2rad(dpitch_deg)
        # Yaw around world Z axis
        Rz = np.array([[ np.cos(yaw), -np.sin(yaw), 0],
                        [ np.sin(yaw),  np.cos(yaw), 0],
                        [ 0,            0,           1]], dtype=np.float64)
        # Pitch around view X axis (left-right axis in current view)
        Rx = np.array([[1, 0,             0           ],
                       [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch),  np.cos(pitch)]], dtype=np.float64)
        self.rot_matrix = Rx @ Rz @ self.rot_matrix
        self._init_view_rotated()

    def _init_view_rotated(self):
        """Re-fit zoom/pan after a rotation so all points stay visible."""
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
        print(f"  Free-rotate mode: {'ON  (arrow keys=rotate, G=exit)' if self.free_rotate else 'OFF (T/F/S views restored)'}")

    def zoom_at(self, px: float, py: float, factor: float):
        """Zoom keeping canvas point (px,py) fixed in world coords."""
        wx, wy   = self.canvas_to_world(px, py)
        self.zoom = max(0.01, self.zoom * factor)
        # adjust pan so (wx,wy) stays under cursor
        new_px, new_py = self.world_to_canvas(wx, wy)
        self.pan[0] += (new_px - px) / self.zoom
        self.pan[1] -= (new_py - py) / self.zoom   # note Y-flip

    def start_pan(self, px: float, py: float):
        self.drag_start = (px, py)
        self.pan_start  = self.pan.copy()

    def update_pan(self, px: float, py: float):
        if self.drag_start is None:
            return
        dx = (px - self.drag_start[0]) / self.zoom
        dy = (py - self.drag_start[1]) / self.zoom
        self.pan[0] = self.pan_start[0] - dx
        self.pan[1] = self.pan_start[1] + dy   # Y-flip

    def end_pan(self):
        self.drag_start = None
        self.pan_start  = None


# ══════════════════════════════════════════════════════════════════════════════
# Camera overlay renderer
# ══════════════════════════════════════════════════════════════════════════════

def draw_camera(img_bgr: np.ndarray, pts2d: list, result=None, pts3d_prior=None) -> np.ndarray:
    canvas = img_bgr.copy()
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

            # skip any NaN/inf or wildly out-of-bounds reprojections
            if not (np.isfinite(pu) and np.isfinite(pv)):
                continue
            pu_i, pv_i = int(np.clip(pu, -10000, 10000)), int(np.clip(pv, -10000, 10000))
            in_bounds = (0 <= pu_i < w) and (0 <= pv_i < h)

            if in_bounds:
                cv2.drawMarker(canvas, (pu_i, pv_i), color,
                               cv2.MARKER_TILTED_CROSS, 14, 2)
            if i < len(pts2d):
                u0, v0 = pts2d[i]
                # line from clicked point to reprojection (clamp endpoint if off-screen)
                pu_c = int(np.clip(pu, 0, w - 1))
                pv_c = int(np.clip(pv, 0, h - 1))
                cv2.line(canvas, (int(u0), int(v0)), (pu_c, pv_c),
                         color, 1, cv2.LINE_AA)
            # error label — place near clicked point so it's always visible
            if i < len(pts2d):
                u0, v0 = pts2d[i]
                cv2.putText(canvas, f'{errs[i]:.1f}px',
                            (int(u0) + 10, int(v0) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # ── Prior reprojection overlay ──────────────────────────────────────────────
    if pts3d_prior:
        try:
            rv_p, t_p = prior_init()
            pts3d_arr = np.array(pts3d_prior, dtype=np.float64)
            proj_p    = project(pts3d_arr, rv_p, t_p)
            h_img, w_img = canvas.shape[:2]
            for i, (pu, pv) in enumerate(proj_p):
                if not (np.isfinite(pu) and np.isfinite(pv)):
                    continue
                color = COLORS_BGR[i % len(COLORS_BGR)]
                pu_i, pv_i = int(np.clip(pu, 0, w_img-1)), int(np.clip(pv, 0, h_img-1))
                in_bounds = (0 <= int(pu) < w_img) and (0 <= int(pv) < h_img)
                # diamond marker for prior projection
                cv2.drawMarker(canvas, (pu_i, pv_i), color,
                               cv2.MARKER_DIAMOND, 20, 2)
                label_x = int(np.clip(pu + 12, 0, w_img - 60))
                label_y = int(np.clip(pv - 12, 12, h_img - 4))
                cv2.putText(canvas, f'P{i+1}{"" if in_bounds else " (off)"}',
                            (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        except Exception:
            pass

    n = len(pts2d)
    hint = '  P=diamond=prior-reproject' if pts3d_prior else ''
    cv2.putText(canvas, f'2D points: {n}  (right-click=remove){hint}',
                (10, canvas.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# Main application
# ══════════════════════════════════════════════════════════════════════════════

class CalibTool:
    def __init__(self, img_bgr: np.ndarray, pts3d: np.ndarray, intensity: np.ndarray):
        self.img      = img_bgr
        self.renderer = LidarRenderer(pts3d, intensity)

        self.pts2d:     list = []
        self.pts3d_sel: list = []
        self.result         = None

        cv2.namedWindow(WINDOW_CAM, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WINDOW_LID, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_CAM, min(self.img.shape[1], 1200),
                                     min(self.img.shape[0],  800))
        cv2.resizeWindow(WINDOW_LID, RENDER_W, RENDER_H)

        cv2.setMouseCallback(WINDOW_CAM, self._cam_mouse)
        cv2.setMouseCallback(WINDOW_LID, self._lid_mouse)

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _refresh(self):
        # Pass 3D picks to camera window so prior reprojection is always shown
        cam_frame = draw_camera(self.img, self.pts2d, self.result,
                                pts3d_prior=self.pts3d_sel if self.pts3d_sel else None)
        lid_frame = self.renderer.render(self.pts3d_sel)
        cv2.imshow(WINDOW_CAM, cam_frame)
        cv2.imshow(WINDOW_LID, lid_frame)

    # ── Mouse callbacks ────────────────────────────────────────────────────────

    def _cam_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts2d.append((float(x), float(y)))
            self.result = None
            self._refresh()
        elif event == cv2.EVENT_RBUTTONDOWN and self.pts2d:
            self.pts2d.pop()
            self.result = None
            self._refresh()

    def _lid_mouse(self, event, x, y, flags, _):
        r = self.renderer
        if event == cv2.EVENT_LBUTTONDOWN:
            pt = r.pick(x, y)
            if pt is not None:
                self.pts3d_sel.append(pt)
                self.result = None
            else:
                print('  No point within click radius — try clicking closer to a dot')
            self._refresh()

        elif event == cv2.EVENT_RBUTTONDOWN and self.pts3d_sel:
            self.pts3d_sel.pop()
            self.result = None
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

    # ── Solver ─────────────────────────────────────────────────────────────────

    def _verify_prior(self):
        """
        Reproject the selected 3D points using the prior transform and print
        where they land — lets you sanity-check picks before solving.
        """
        if not self.pts3d_sel:
            print('  No 3D points selected yet')
            return
        rv, t = prior_init()
        pts3d = np.array(self.pts3d_sel, dtype=np.float64)
        proj  = project(pts3d, rv, t)
        h, w  = self.img.shape[:2]
        print('\n  Prior reprojection of your 3D picks onto camera image:')
        for i, (u, v) in enumerate(proj):
            in_frame = (0 <= u < w) and (0 <= v < h)
            flag = '' if in_frame else '  ← OFF SCREEN'
            print(f'    Point {i+1}: ({u:.0f}, {v:.0f}){flag}')
        if self.pts2d:
            print('  Your 2D picks:')
            for i, (u, v) in enumerate(self.pts2d):
                print(f'    Point {i+1}: ({u:.0f}, {v:.0f})')
        print('  (If prior reprojects near your 2D picks, your 3D picks are good)')

    def _run_solver(self):
        n2, n3 = len(self.pts2d), len(self.pts3d_sel)
        if n2 != n3:
            print(f'  ✗ Cannot solve: {n2} 2D pts vs {n3} 3D pts — counts must match')
            return
        if n2 < 3:
            print(f'  ✗ Need at least 3 pairs (have {n2})')
            return

        pts2d = np.array(self.pts2d,     dtype=np.float64)
        pts3d = np.array(self.pts3d_sel, dtype=np.float64)

        print('\n── Running Algorithm 1 (Koide ICRA 2023) ────────────────────')
        try:
            self.result = solve(pts2d, pts3d)
        except Exception as e:
            print(f'  Solver error: {e}')
            return

        R, t = self.result['R'], self.result['t']
        errs = self.result['reproj_errors']
        T    = np.eye(4); T[:3, :3] = R; T[:3, 3] = t

        print(f'  RANSAC inliers : {self.result["inlier_mask"].sum()} / {n2}')
        print(f'  Reprojection errors (px): {errs.round(2)}')
        print(f'  Mean error: {errs.mean():.3f} px   Max: {errs.max():.3f} px')
        print(f'\n  4×4 Transform T_cam_lidar (LiDAR → Camera):\n{np.round(T, 6)}')
        print('─────────────────────────────────────────────────────────────\n')
        self._refresh()

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self):
        self._refresh()
        print('\nReady.  Click points, then press ENTER to solve.\n')

        while True:
            key = cv2.waitKeyEx(20)

            if key in (13, 10):           # ENTER
                self._run_solver()
            elif key == ord('r'):
                self.pts2d.clear()
                self.pts3d_sel.clear()
                self.result = None
                self._refresh()
            elif key in (ord('v'), ord('V')):
                self._verify_prior()
            elif key in (ord('t'), ord('T')):
                self.renderer.set_view('T'); self._refresh()
            elif key in (ord('f'), ord('F')):
                self.renderer.set_view('F'); self._refresh()
            elif key in (ord('s'), ord('S')):
                self.renderer.set_view('S'); self._refresh()
            elif key in (ord('g'), ord('G')):
                self.renderer.toggle_free_rotate(); self._refresh()
            elif key in (65361, 2424832):   # left arrow  (Linux / Windows)
                self.renderer.rotate_by(dyaw_deg=-5, dpitch_deg=0); self._refresh()
            elif key in (65363, 2555904):   # right arrow
                self.renderer.rotate_by(dyaw_deg=+5, dpitch_deg=0); self._refresh()
            elif key in (65362, 2490368):   # up arrow
                self.renderer.rotate_by(dyaw_deg=0, dpitch_deg=-5); self._refresh()
            elif key in (65364, 2621440):   # down arrow
                self.renderer.rotate_by(dyaw_deg=0, dpitch_deg=+5); self._refresh()
            elif key in (ord('q'), ord('Q'), 27):   # Q or ESC
                break

            # also quit if either window was closed
            if cv2.getWindowProperty(WINDOW_CAM, cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.getWindowProperty(WINDOW_LID, cv2.WND_PROP_VISIBLE) < 1:
                break
        if self.result is not None:
            T = np.eye(4)
            T[:3,:3] = self.result['R']
            T[:3, 3] = self.result['t']
            np.save('calibration_result.npy', T)
            print(f"\nResult saved to calibration_result.npy")
            print(f"T_cam_lidar:\n{np.round(T,6)}")
            print(f"Human Readable Rotation vector (x,y,z): {Rotation.from_matrix(self.result['R']).as_rotvec()}")
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Manual LiDAR-Camera calibration (Koide Algorithm 1, ICRA 2023)')
    parser.add_argument('camera_image', help='Camera image (.png / .jpg)')
    parser.add_argument('lidar_npy',    help='Blickfeld LiDAR .npy file')
    args = parser.parse_args()

    print(f'Loading camera image : {args.camera_image}')
    img = load_camera(args.camera_image)
    print(f'  {img.shape[1]} × {img.shape[0]} px')

    print(f'Loading LiDAR data   : {args.lidar_npy}')
    pts3d, intensity = load_lidar(args.lidar_npy)
    print(f'  {len(pts3d):,} points  (rendering up to {MAX_POINTS:,})')

    CalibTool(img, pts3d, intensity).run()
    

if __name__ == '__main__':
    main()