#!/usr/bin/env python3
"""
Target Point-Cloud Labeler  (manual prototype)
--------------------------------------------------------------
Prototype for an automatic point-cloud labeling pipeline: project LiDAR
points into a camera image using a calibrated transform, select a target
region in the image (human, etc.), and label the LiDAR points that fall on
that target — while rejecting points that are in the same 2D box/mask but
sit behind the target (e.g. a wall) using depth clustering.

This script is the manual verification harness for that idea: the region
selector is either a hand-drawn box or a loaded mask image, so a future
automatic version just needs to swap in a segmentation model's output mask
in place of --mask (or the interactive box) — nothing else changes.

──────────────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────────────
  python label_target.py <camera_image> <lidar_npy> <transform.json> \\
      [--mask mask.png] [--intrinsics intrinsics.json] \\
      [--target-fraction 0.7] [--depth-padding 0.3] \\
      [--out-prefix ./label]

Arguments
---------
  camera_image     Camera image (.png / .jpg)
  lidar_npy        Blickfeld LiDAR .npy file
  transform.json   Calibration result from manual_calibrate.py
                    (translation + quaternion, optionally c / delta)

Depth discrimination
---------------------
Within the selected image region, LiDAR points can straddle both the target
and whatever is behind it (a wall, another person, background clutter).
This script separates them with a hybrid method:
  1. Gap-based (primary): sort the in-region depths and look for the
     largest gap between consecutive points. If it's at least
     --gap-threshold metres wide, split there and keep the cluster nearest
     the camera (the target is assumed to be in front of the background).
  2. Fraction-based (fallback): if no clean gap is found (target and
     background blend together), falls back to the narrowest depth window
     containing --target-fraction of the in-region points.
Points within the detected window (+/- --depth-padding) are kept as target;
everything else in the region is treated as background and excluded.

Options
-------
  --mask FILE       Binary segmentation mask (same size as camera_image;
                     nonzero = target). If given, skips the interactive box
                     and uses the mask directly — this is the hook for an
                     automatic CV model.
  --intrinsics FILE Camera intrinsics JSON (overrides hardcoded defaults;
                     may also embed translation/quaternion — see
                     visualize_calibration.py for the schema).
  --target-fraction Fallback fraction used only when no clean depth gap is
                     found. Default 0.7 (i.e. 70%).
  --depth-padding   Metres of slack added around the detected target depth
                     window before rejecting a point as background.
                     Default 0.3 m.
  --gap-threshold   Minimum depth gap (m) between clusters to treat as a
                     real target/background split. Default 0.4 m.
  --out-prefix      Output path prefix for saved files. Default './label'.

Outputs  (all written on quit)
-------------------------------
  <out-prefix>_camera_overlay.png   camera image + box + classified points
  <out-prefix>_pointcloud_<V>.png   point-cloud view exactly as it looked
                                    when you quit (V = T/F/S — whichever
                                    view was active)
  <out-prefix>_combined.png         the two above, side by side
  <out-prefix>_label.json           target point indices + metadata

Point-cloud window controls
----------------------------
  T / F / S       Top / Front / Side view
  Scroll          Zoom
  Middle-drag     Pan
  B               Re-draw the box (manual box mode only)
  [ / ]           Decrease / increase depth padding by 0.05 m
  - / =           Decrease / increase gap threshold by 0.05 m
  R               Recompute with current padding / gap threshold
  Q / ESC         Quit — saves all three images + the label JSON
"""

import argparse
import json
import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# ── Default camera intrinsics (used when no --intrinsics JSON is given) ───────
_DEFAULT_SCALE = 1
_DEFAULT_FX    = 1248.8 * _DEFAULT_SCALE
_DEFAULT_FY    = 1244.6 * _DEFAULT_SCALE
_DEFAULT_CX    = 945.08 * _DEFAULT_SCALE
_DEFAULT_CY    = 527.51 * _DEFAULT_SCALE
_DEFAULT_DIST  = [0.1949, -0.3245, 0.0, 0.0, 0.0]

FX, FY, CX, CY = _DEFAULT_FX, _DEFAULT_FY, _DEFAULT_CX, _DEFAULT_CY
DIST = np.array(_DEFAULT_DIST, dtype=np.float64)
K    = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)

RENDER_W, RENDER_H = 900, 700
MAX_POINTS = 300_000

COLOR_TARGET     = (60, 220, 60)     # BGR bright green
COLOR_BACKGROUND = (0, 140, 255)     # BGR orange
COLOR_CONTEXT    = None              # grayscale by intensity


# ══════════════════════════════════════════════════════════════════════════════
# Intrinsics / transform loading  (same schema as visualize_calibration.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_intrinsics(path: str) -> dict:
    global FX, FY, CX, CY, DIST, K
    with open(path) as f:
        data = json.load(f)
    required = {'fx', 'fy', 'cx', 'cy', 'dist_coeffs'}
    missing = required - data.keys()
    if missing:
        raise ValueError(f'Intrinsics JSON missing keys: {missing}')
    FX, FY, CX, CY = (float(data['fx']), float(data['fy']),
                       float(data['cx']), float(data['cy']))
    DIST = np.array(data['dist_coeffs'], dtype=np.float64)
    K    = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
    print(f'  Intrinsics: fx={FX:.2f} fy={FY:.2f} cx={CX:.2f} cy={CY:.2f}')
    return data


def load_transform(path: str):
    """Returns (T, c, delta) — same schema written by manual_calibrate.py."""
    with open(path) as f:
        data = json.load(f)
    if 'translation' not in data or 'quaternion' not in data:
        raise ValueError(
            f'Transform JSON "{path}" must contain "translation" and '
            f'"quaternion" keys.  Found: {list(data.keys())}')
    t = np.array(data['translation'], dtype=np.float64)
    R = Rotation.from_quat(data['quaternion']).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = t
    c     = float(data.get('c', 1.0))
    delta = float(data.get('delta', 1.0))
    if c != 1.0 or delta != 1.0:
        print(f'  Scale corrections found — c: {c:.5f}   delta: {delta:.5f}')
    return T, c, delta


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
    return cart[valid], intensity[valid]


# ══════════════════════════════════════════════════════════════════════════════
# Projection  (same c/delta model as the other scripts)
# ══════════════════════════════════════════════════════════════════════════════

def project_points(pts3d: np.ndarray, T: np.ndarray,
                   c: float = 1.0, delta: float = 1.0):
    """
    Returns px, py (pixel coords, NaN if behind camera), depth (camera-frame
    Z, metres), in_front (bool mask).
    """
    pts_eff = pts3d * delta if delta != 1.0 else pts3d
    R, t    = T[:3, :3], T[:3, 3]
    cam     = (R @ pts_eff.T).T + t

    depth    = cam[:, 2]
    in_front = depth > 0.01

    px = np.full(len(pts3d), np.nan)
    py = np.full(len(pts3d), np.nan)
    if in_front.sum() == 0:
        return px, py, depth, in_front

    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    tvec    = t.astype(np.float64).reshape(3, 1)
    K_eff   = K
    if c != 1.0:
        K_eff = K.copy()
        K_eff[0, 0] *= c
        K_eff[1, 1] *= c

    proj, _ = cv2.projectPoints(pts_eff[in_front].astype(np.float64),
                                rvec, tvec, K_eff, DIST)
    proj = proj.reshape(-1, 2)
    px[in_front] = proj[:, 0]
    py[in_front] = proj[:, 1]
    return px, py, depth, in_front


# ══════════════════════════════════════════════════════════════════════════════
# Region selection — manual box or a loaded mask
# ══════════════════════════════════════════════════════════════════════════════

def select_box_interactive(img: np.ndarray):
    """Draw a rectangle over the target. Returns (x, y, w, h) or None."""
    win = 'Draw box around target  [ENTER/SPACE=confirm  C=cancel]'
    x, y, w, h = cv2.selectROI(win, img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win)
    if w == 0 or h == 0:
        return None
    return int(x), int(y), int(w), int(h)


def region_mask_from_box(img_shape, box) -> np.ndarray:
    h_img, w_img = img_shape[:2]
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    x, y, w, h = box
    mask[y:y + h, x:x + w] = 255
    return mask


def load_mask_file(path: str, img_shape) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f'Cannot read mask: {path}')
    if m.shape[:2] != img_shape[:2]:
        raise ValueError(
            f'Mask size {m.shape[:2]} does not match image size {img_shape[:2]}')
    return m


# ══════════════════════════════════════════════════════════════════════════════
# Depth discrimination — separate target from whatever is behind it
# ══════════════════════════════════════════════════════════════════════════════

def shortest_depth_window(depths: np.ndarray, fraction: float = 0.7):
    """
    Find the narrowest depth interval containing at least `fraction` of the
    given depths — the "highest density interval". This locks onto whichever
    depth range is densest (the target, if it's most of the region) rather
    than being pulled around by a background spike (e.g. a wall), unlike a
    plain mean/std or min/max approach.

    Used as the fallback inside hybrid_depth_window() when no clean gap is
    found — see that function for why a fraction alone isn't reliable.

    Returns (lo, hi) or None if depths is empty.
    """
    if len(depths) == 0:
        return None
    d = np.sort(depths)
    n = len(d)
    m = max(1, int(np.ceil(fraction * n)))
    if m >= n:
        return float(d[0]), float(d[-1])
    widths = d[m - 1:] - d[:n - m + 1]
    best = int(np.argmin(widths))
    return float(d[best]), float(d[best + m - 1])


def hybrid_depth_window(depths: np.ndarray, target_fraction: float = 0.7,
                        gap_threshold: float = 0.4):
    """
    Hybrid target/background depth-window detector.

    1) Gap-based (primary): sort the in-region depths and find the largest
       gap between consecutive points. If that gap is at least
       gap_threshold metres wide, split there and treat the NEAREST cluster
       (closest to the camera) as the target — matching the physical setup
       described: the target is in front of whatever is behind it (a wall,
       clutter). This doesn't need to know what fraction of the box is
       actually target, which is the failure mode of a pure-fraction
       approach: if the box is loose enough that the target is well under
       target_fraction of the points, a fixed-fraction window is forced to
       bridge the gap and swallow the background cluster whole.

    2) Fraction-based (fallback): if no gap that large exists — target and
       background blend together with no clean separation (e.g. the person
       is standing flush against the wall) — falls back to the "shortest
       window containing target_fraction of points" method.

    Returns (lo, hi, method) where method is 'gap' or 'fraction', or None
    if depths is empty.
    """
    if len(depths) == 0:
        return None
    d = np.sort(depths)
    if len(d) == 1:
        return float(d[0]), float(d[0]), 'gap'

    gaps = np.diff(d)
    gi = int(np.argmax(gaps))
    if gaps[gi] >= gap_threshold:
        near = d[:gi + 1]     # nearest-to-camera cluster = assumed target
        return float(near[0]), float(near[-1]), 'gap'

    window = shortest_depth_window(d, fraction=target_fraction)
    return window[0], window[1], 'fraction'


def classify_points(pts3d: np.ndarray, T: np.ndarray, c: float, delta: float,
                    region_mask: np.ndarray, img_shape,
                    target_fraction: float = 0.7, depth_padding: float = 0.3,
                    gap_threshold: float = 0.4):
    """
    Returns a dict with:
      px, py, depth, in_front   — raw projection results (for rendering)
      in_region                 — bool mask, projects inside region_mask
      target_mask                — in_region AND within the detected depth window
      background_mask             — in_region but rejected on depth
      depth_window                — (lo, hi, lo_padded, hi_padded) or None
      depth_window_method         — 'gap' or 'fraction', or None
    """
    px, py, depth, in_front = project_points(pts3d, T, c=c, delta=delta)
    h_img, w_img = img_shape[:2]

    px_i = np.clip(np.nan_to_num(px, nan=-1), -1, w_img)
    py_i = np.clip(np.nan_to_num(py, nan=-1), -1, h_img)
    in_bounds = (in_front & np.isfinite(px) & np.isfinite(py) &
                (px >= 0) & (px < w_img) & (py >= 0) & (py < h_img))

    in_region = np.zeros(len(pts3d), dtype=bool)
    if in_bounds.sum() > 0:
        xi = px_i[in_bounds].astype(np.int32)
        yi = py_i[in_bounds].astype(np.int32)
        hit = region_mask[yi, xi] > 0
        idx = np.where(in_bounds)[0]
        in_region[idx[hit]] = True

    if in_region.sum() == 0:
        return dict(px=px, py=py, depth=depth, in_front=in_front,
                   in_region=in_region,
                   target_mask=in_region.copy(),
                   background_mask=np.zeros_like(in_region),
                   depth_window=None, depth_window_method=None)

    lo, hi, method = hybrid_depth_window(depth[in_region],
                                        target_fraction=target_fraction,
                                        gap_threshold=gap_threshold)
    lo_p, hi_p = lo - depth_padding, hi + depth_padding
    depth_ok = (depth >= lo_p) & (depth <= hi_p)

    target_mask     = in_region & depth_ok
    background_mask = in_region & ~depth_ok

    return dict(px=px, py=py, depth=depth, in_front=in_front,
               in_region=in_region, target_mask=target_mask,
               background_mask=background_mask,
               depth_window=(lo, hi, lo_p, hi_p),
               depth_window_method=method)


# ══════════════════════════════════════════════════════════════════════════════
# Point-cloud viewer  (top/front/side, zoom/pan — same conventions as
# manual_calibrate.py's LidarRenderer, trimmed down)
# ══════════════════════════════════════════════════════════════════════════════

VIEWS = {
    'T': (0, 1, 'TOP-DOWN    (X right, Y forward)'),
    'F': (0, 2, 'FRONT-FACE  (X right, Z up)'),
    'S': (1, 2, 'SIDE        (Y forward, Z up)'),
}


class LabelViewer:
    def __init__(self, pts3d, intensity, result):
        self.pts3d     = pts3d
        ni = intensity.astype(np.float32)
        lo, hi = np.percentile(ni, 2), np.percentile(ni, 98)
        ni = np.clip((ni - lo) / max(hi - lo, 1e-6), 0, 1)
        self.intensity_u8 = (ni * 120).astype(np.uint8)   # dim, so labels pop

        step = max(1, len(pts3d) // MAX_POINTS)
        self.render_idx = np.arange(0, len(pts3d), step)

        self.result   = result
        self.view_key = 'T'
        self.zoom     = 1.0
        self.pan      = np.array([0.0, 0.0])
        self._init_view()

        self.drag_start = None
        self.pan_start  = None

    def _init_view(self):
        ax, ay, _ = VIEWS[self.view_key]
        xs = self.pts3d[self.render_idx, ax]
        ys = self.pts3d[self.render_idx, ay]
        cx, cy = xs.mean(), ys.mean()
        span   = max(np.ptp(xs), np.ptp(ys), 1e-3)
        self.zoom = min(RENDER_W, RENDER_H) / span * 0.85
        self.pan  = np.array([cx, cy])

    def set_view(self, key):
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

    def render(self) -> np.ndarray:
        canvas = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
        ax, ay, label = VIEWS[self.view_key]

        idx = self.render_idx
        sub = self.pts3d[idx]
        px, py = self.world_to_canvas(sub[:, ax], sub[:, ay])
        in_view = (px >= 0) & (px < RENDER_W) & (py >= 0) & (py < RENDER_H)

        target_m     = self.result['target_mask'][idx]
        background_m = self.result['background_mask'][idx]
        context_m    = ~(target_m | background_m)

        def scatter(mask, color_fn):
            sel = mask & in_view
            xs = px[sel].astype(np.int32)
            ys = py[sel].astype(np.int32)
            cols = color_fn(sel)
            for i in range(len(xs)):
                c = cols[i] if isinstance(cols, np.ndarray) else cols
                cv2.circle(canvas, (xs[i], ys[i]), 1,
                          (int(c[0]), int(c[1]), int(c[2])), -1)

        # Context cloud first (dim, intensity-shaded), then background,
        # then target on top so it's never occluded.
        scatter(context_m, lambda sel: np.stack(
            [self.intensity_u8[idx][sel]] * 3, axis=1))
        scatter(background_m, lambda sel: COLOR_BACKGROUND)
        scatter(target_m, lambda sel: COLOR_TARGET)

        n_t = int(self.result['target_mask'].sum())
        n_b = int(self.result['background_mask'].sum())
        dw  = self.result['depth_window']
        method = self.result.get('depth_window_method')
        dw_str = (f'window: {dw[0]:.2f}-{dw[1]:.2f}m  [{method}]  '
                 f'(+/-pad -> {dw[2]:.2f}-{dw[3]:.2f}m)') if dw else 'window: n/a'

        cv2.putText(canvas, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                   (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, f'target: {n_t}  background(rejected): {n_b}',
                   (10, RENDER_H - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                   COLOR_TARGET, 1, cv2.LINE_AA)
        cv2.putText(canvas, dw_str, (10, RENDER_H - 14),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        return canvas

    def zoom_at(self, px, py, factor):
        wx, wy = self.canvas_to_world(px, py)
        self.zoom = max(0.01, self.zoom * factor)
        npx, npy = self.world_to_canvas(wx, wy)
        self.pan[0] += (npx - px) / self.zoom
        self.pan[1] -= (npy - py) / self.zoom

    def start_pan(self, px, py):
        self.drag_start = (px, py)
        self.pan_start  = self.pan.copy()

    def update_pan(self, px, py):
        if self.drag_start is None:
            return
        dx = (px - self.drag_start[0]) / self.zoom
        dy = (py - self.drag_start[1]) / self.zoom
        self.pan = self.pan_start - np.array([dx, -dy])

    def end_pan(self):
        self.drag_start = None


def build_camera_overlay(img: np.ndarray, box, result) -> np.ndarray:
    """Camera image with the box (if any) and classified points drawn on it."""
    overlay = img.copy()
    if box:
        x, y, w, h = box
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)
    px, py = result['px'], result['py']
    for mask, color in [(result['background_mask'], COLOR_BACKGROUND),
                        (result['target_mask'], COLOR_TARGET)]:
        xs = px[mask]; ys = py[mask]
        for u, v in zip(xs, ys):
            if np.isfinite(u) and np.isfinite(v):
                cv2.circle(overlay, (int(u), int(v)), 2, color, -1)
    return overlay


def build_combined_image(camera_img: np.ndarray, pc_img: np.ndarray) -> np.ndarray:
    """Side-by-side composite of the camera overlay and point-cloud view."""
    h = max(camera_img.shape[0], pc_img.shape[0])

    def resize_to_h(im):
        scale = h / im.shape[0]
        return cv2.resize(im, (int(im.shape[1] * scale), h))

    left  = resize_to_h(camera_img)
    right = resize_to_h(pc_img)
    return np.hstack([left, right])


# ══════════════════════════════════════════════════════════════════════════════
# Saving
# ══════════════════════════════════════════════════════════════════════════════

def save_label(out_prefix, image_path, lidar_path, transform_path,
              region_mode, box, mask_path, result,
              target_fraction, depth_padding, gap_threshold,
              image_paths):
    idx = np.where(result['target_mask'])[0]
    out = {
        'source_image':     image_path,
        'source_lidar':      lidar_path,
        'source_transform':  transform_path,
        'region_mode':       region_mode,          # 'box' or 'mask'
        'box':               list(box) if box else None,
        'mask_file':         mask_path,
        'target_fraction':   target_fraction,
        'depth_padding':     depth_padding,
        'gap_threshold':     gap_threshold,
        'depth_window':      list(result['depth_window']) if result['depth_window'] else None,
        'depth_window_method': result.get('depth_window_method'),
        'n_target_points':   int(len(idx)),
        'n_background_rejected': int(result['background_mask'].sum()),
        'target_indices':    idx.tolist(),
        'saved_images':      image_paths,
    }
    out_path = f'{out_prefix}_label.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {len(idx)} target point indices -> {out_path}')
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Manual target point-cloud labeler (box/mask + depth split)',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('camera_image')
    parser.add_argument('lidar_npy')
    parser.add_argument('transform')
    parser.add_argument('--mask', default=None, metavar='FILE',
                        help='Binary segmentation mask instead of a drawn box')
    parser.add_argument('--intrinsics', default=None, metavar='FILE')
    parser.add_argument('--target-fraction', type=float, default=0.7)
    parser.add_argument('--depth-padding', type=float, default=0.3)
    parser.add_argument('--gap-threshold', type=float, default=0.4,
                        help='Minimum depth gap (m) between clusters to treat '
                             'as a real target/background split. Below this, '
                             'falls back to the target-fraction method.')
    parser.add_argument('--out-prefix', default='./label')
    args = parser.parse_args()

    if args.intrinsics:
        print(f'Loading intrinsics: {args.intrinsics}')
        load_intrinsics(args.intrinsics)

    print(f'Loading image  : {args.camera_image}')
    img = cv2.imread(args.camera_image)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {args.camera_image}')
    print(f'  {img.shape[1]} x {img.shape[0]} px')

    print(f'Loading LiDAR  : {args.lidar_npy}')
    pts3d, intensity = load_lidar(args.lidar_npy)
    print(f'  {len(pts3d):,} points')

    print(f'Loading transform: {args.transform}')
    T, c, delta = load_transform(args.transform)

    # ── Region selection ───────────────────────────────────────────────────
    box = None
    if args.mask:
        print(f'Loading mask: {args.mask}')
        region_mask = load_mask_file(args.mask, img.shape)
        region_mode = 'mask'
    else:
        box = select_box_interactive(img)
        if box is None:
            print('No box drawn — exiting.')
            return
        region_mask = region_mask_from_box(img.shape, box)
        region_mode = 'box'

    result = classify_points(pts3d, T, c, delta, region_mask, img.shape,
                             target_fraction=args.target_fraction,
                             depth_padding=args.depth_padding,
                             gap_threshold=args.gap_threshold)
    print(f'  In-region points : {int(result["in_region"].sum())}')
    print(f'  Target (kept)    : {int(result["target_mask"].sum())}')
    print(f'  Background (rejected by depth): {int(result["background_mask"].sum())}')
    if result['depth_window']:
        lo, hi, lo_p, hi_p = result['depth_window']
        print(f'  Detected target depth window: {lo:.2f}-{hi:.2f} m '
              f'(padded: {lo_p:.2f}-{hi_p:.2f} m)  method={result["depth_window_method"]}')

    # ── Camera-window overlay (static, for reference) ─────────────────────
    overlay = build_camera_overlay(img, box, result)
    win_cam = 'Camera overlay  [any key = open point-cloud view]'
    cv2.namedWindow(win_cam, cv2.WINDOW_NORMAL)
    cv2.imshow(win_cam, overlay)
    cv2.waitKey(0)
    cv2.destroyWindow(win_cam)

    # ── Point-cloud viewer ──────────────────────────────────────────────────
    viewer = LabelViewer(pts3d, intensity, result)
    win_pc = ('Target Labels  [T/F/S=view  scroll=zoom  mid-drag=pan  '
             'B=redraw box  [/]=depth pad  -/==gap thresh  R=recompute  '
             'Q=quit+save]')
    cv2.namedWindow(win_pc, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_pc, RENDER_W, RENDER_H)

    depth_padding = args.depth_padding
    gap_threshold = args.gap_threshold

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            factor = 1.15 if flags > 0 else 1 / 1.15
            viewer.zoom_at(mx, my, factor)
        elif event == cv2.EVENT_MBUTTONDOWN:
            viewer.start_pan(mx, my)
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_MBUTTON:
            viewer.update_pan(mx, my)
        elif event == cv2.EVENT_MBUTTONUP:
            viewer.end_pan()

    cv2.setMouseCallback(win_pc, on_mouse)

    while True:
        cv2.imshow(win_pc, viewer.render())
        key = cv2.waitKeyEx(20)

        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('t'), ord('T')):
            viewer.set_view('T')
        elif key in (ord('f'), ord('F')):
            viewer.set_view('F')
        elif key in (ord('s'), ord('S')):
            viewer.set_view('S')
        elif key == ord('['):
            depth_padding = max(0.0, depth_padding - 0.05)
            print(f'  depth padding -> {depth_padding:.2f} m  (press R to apply)')
        elif key == ord(']'):
            depth_padding += 0.05
            print(f'  depth padding -> {depth_padding:.2f} m  (press R to apply)')
        elif key == ord('-'):
            gap_threshold = max(0.0, gap_threshold - 0.05)
            print(f'  gap threshold -> {gap_threshold:.2f} m  (press R to apply)')
        elif key == ord('='):
            gap_threshold += 0.05
            print(f'  gap threshold -> {gap_threshold:.2f} m  (press R to apply)')
        elif key in (ord('r'), ord('R')):
            result = classify_points(pts3d, T, c, delta, region_mask, img.shape,
                                     target_fraction=args.target_fraction,
                                     depth_padding=depth_padding,
                                     gap_threshold=gap_threshold)
            viewer.result = result
            print(f'  Recomputed — target: {int(result["target_mask"].sum())}  '
                  f'background: {int(result["background_mask"].sum())}  '
                  f'method={result["depth_window_method"]}')
        elif key in (ord('b'), ord('B')) and region_mode == 'box':
            new_box = select_box_interactive(img)
            if new_box is not None:
                box = new_box
                region_mask = region_mask_from_box(img.shape, box)
                result = classify_points(pts3d, T, c, delta, region_mask, img.shape,
                                         target_fraction=args.target_fraction,
                                         depth_padding=depth_padding,
                                         gap_threshold=gap_threshold)
                viewer.result = result

        if cv2.getWindowProperty(win_pc, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

    # ── Save all three images: camera overlay, point-cloud view exactly as
    #    it looked when the window was closed (whichever T/F/S it was on,
    #    current zoom/pan), and a combined side-by-side for quick review ──
    final_overlay = build_camera_overlay(img, box, result)
    final_pc      = viewer.render()
    final_combined = build_combined_image(final_overlay, final_pc)

    path_overlay  = f'{args.out_prefix}_camera_overlay.png'
    path_pc       = f'{args.out_prefix}_pointcloud_{viewer.view_key}.png'
    path_combined = f'{args.out_prefix}_combined.png'
    cv2.imwrite(path_overlay, final_overlay)
    cv2.imwrite(path_pc, final_pc)
    cv2.imwrite(path_combined, final_combined)
    print(f'\nSaved images:')
    print(f'  camera overlay : {path_overlay}')
    print(f'  point cloud ({viewer.view_key}) : {path_pc}')
    print(f'  combined        : {path_combined}')

    save_label(args.out_prefix, args.camera_image, args.lidar_npy,
              args.transform, region_mode, box, args.mask, result,
              args.target_fraction, depth_padding, gap_threshold,
              image_paths={'camera_overlay': path_overlay,
                          'point_cloud': path_pc,
                          'combined': path_combined})


if __name__ == '__main__':
    main()