#!/usr/bin/env python3
"""
LiDAR-Camera Calibration Visualiser
-------------------------------------
Two outputs from a saved T_cam_lidar transform:

  1. lidar_on_image  — LiDAR points projected onto the camera image,
                       coloured by depth (close=red, far=blue)

  2. image_on_lidar  — Each LiDAR point coloured by the RGB value of the
                       camera pixel it maps to (front view).
                       Points that fall outside the image are grey.

──────────────────────────────────────────────────────────────────────────────
SINGLE-FILE MODE
──────────────────────────────────────────────────────────────────────────────
Usage:
  python visualize_calibration.py <camera_image> <lidar_npy> <transform>

Arguments:
  camera_image   Path to camera image (.png / .jpg)
  lidar_npy      Path to Blickfeld LiDAR .npy file
  transform      4×4 T_cam_lidar matrix (.npy)  OR  calibration JSON (see below)

──────────────────────────────────────────────────────────────────────────────
DIRECTORY MODE
──────────────────────────────────────────────────────────────────────────────
Usage:
  python visualize_calibration.py --dir <directory>

The tool scans <directory> for matched triplets of files:
  • camera image  : any  .png / .jpg  whose stem contains "image" or "cam"
  • LiDAR cloud   : any  .npy         whose stem contains "lidar" or "cloud"
  • transform     : any  .npy / .json whose stem contains "transform",
                    "calibration", or "calib"

Each matched triplet is processed and outputs are saved in a sub-directory
named after the run (e.g. <directory>/run_001/).

──────────────────────────────────────────────────────────────────────────────
JSON FORMATS
──────────────────────────────────────────────────────────────────────────────
Intrinsics JSON  (--intrinsics path/to/intrinsics.json):
  {
    "fx": 2497.6,
    "fy": 2489.2,
    "cx": 1890.16,
    "cy": 1055.02,
    "dist_coeffs": [0.1949, -0.3245, 0.0, 0.0, 0.0]
  }

Transform JSON  (pass as <transform> argument or found during directory scan):
  {
    "translation": [tx, ty, tz],
    "quaternion":  [qx, qy, qz, qw]
  }

  An intrinsics JSON may also embed a transform under the same keys, letting
  you use a single file for everything:
  {
    "fx": ..., "fy": ..., "cx": ..., "cy": ..., "dist_coeffs": [...],
    "translation": [tx, ty, tz],
    "quaternion":  [qx, qy, qz, qw]
  }

──────────────────────────────────────────────────────────────────────────────
Optional flags:
  --dir DIR      Process all matched triplets in a directory
  --intrinsics F Path to camera intrinsics JSON (overrides hardcoded values)
  --no-display   Skip interactive windows, only save output images
  --out-dir DIR  Root directory for output images
                   single-file mode : default = directory of transform file
                   directory mode   : default = <dir>/visualizations/
  --max-depth M  Clip depth colour scale at M metres (default: auto)
  --radius R     Point radius in pixels for projected points (default: 2)
"""

import argparse
import json
import os
import sys
import glob
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

# Module-level intrinsic globals (overwritten by load_intrinsics / set_intrinsics)
FX   = _DEFAULT_FX
FY   = _DEFAULT_FY
CX   = _DEFAULT_CX
CY   = _DEFAULT_CY
DIST = np.array(_DEFAULT_DIST, dtype=np.float64)
K    = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)

MAX_POINTS = 500_000   # downsample above this

# ── LiDAR front-view render settings ─────────────────────────────────────────
RENDER_W = 1200
RENDER_H = 900


# ══════════════════════════════════════════════════════════════════════════════
# Intrinsics helpers
# ══════════════════════════════════════════════════════════════════════════════

def set_intrinsics(fx: float, fy: float, cx: float, cy: float,
                   dist_coeffs: list) -> None:
    """Update the module-level camera intrinsic globals."""
    global FX, FY, CX, CY, DIST, K
    FX   = float(fx)
    FY   = float(fy)
    CX   = float(cx)
    CY   = float(cy)
    DIST = np.array(dist_coeffs, dtype=np.float64)
    K    = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
    print(f'  Intrinsics set: fx={FX:.2f}  fy={FY:.2f}  cx={CX:.2f}  cy={CY:.2f}')
    print(f'  Distortion    : {DIST.tolist()}')


def load_intrinsics(path: str) -> dict:
    """
    Load camera intrinsics from a JSON file and apply them globally.
    Required keys: fx, fy, cx, cy, dist_coeffs
    Optional keys: translation, quaternion  (parsed but not applied here)
    Returns the raw dict for downstream use (e.g. embedded transform).
    """
    with open(path) as f:
        data = json.load(f)
    required = {'fx', 'fy', 'cx', 'cy', 'dist_coeffs'}
    missing = required - data.keys()
    if missing:
        raise ValueError(f'Calibration JSON missing keys: {missing}')
    set_intrinsics(data['fx'], data['fy'], data['cx'], data['cy'],
                   data['dist_coeffs'])
    return data


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


def _quat_trans_to_matrix(translation, quaternion) -> np.ndarray:
    """
    Build a 4×4 homogeneous transform from translation [tx,ty,tz] and
    quaternion [qx, qy, qz, qw].
    """
    t = np.array(translation, dtype=np.float64)
    R = Rotation.from_quat(quaternion).as_matrix()   # scipy: [qx,qy,qz,qw]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


def load_transform(path: str) -> np.ndarray:
    """
    Load a 4×4 T_cam_lidar matrix from:
      • a .npy file containing a (4,4) array, OR
      • a .json file with keys "translation" and "quaternion".
    An intrinsics JSON that also contains those keys is handled transparently.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == '.npy':
        T = np.load(path)
        assert T.shape == (4, 4), f'Expected 4×4 matrix, got {T.shape}'
        return T.astype(np.float64)

    if ext == '.json':
        with open(path) as f:
            data = json.load(f)
        if 'translation' not in data or 'quaternion' not in data:
            raise ValueError(
                f'Transform JSON "{path}" must contain "translation" and "quaternion" keys.\n'
                f'  Found keys: {list(data.keys())}')
        T = _quat_trans_to_matrix(data['translation'], data['quaternion'])
        #print(f'  Loaded transform from quaternion:')
        #print(f'    translation : {data["translation"]}')
        #print(f'    quaternion  : {data["quaternion"]}  (xyzw)')
        return T

    raise ValueError(f'Unsupported transform file extension "{ext}". Use .npy or .json.')


# ══════════════════════════════════════════════════════════════════════════════
# Projection helpers
# ══════════════════════════════════════════════════════════════════════════════

def project_points(pts3d: np.ndarray, T: np.ndarray):
    """
    Project Nx3 LiDAR points into the camera image using T_cam_lidar.
    Returns:
      px, py  : (N,) pixel coordinates (float)
      depth   : (N,) depth in camera Z axis (metres)
      in_front: (N,) bool mask — points with positive depth
    """
    R   = T[:3, :3]
    t   = T[:3,  3]
    cam = (R @ pts3d.T).T + t          # (N,3) in camera frame

    depth    = cam[:, 2]
    in_front = depth > 0.01

    px = np.full(len(pts3d), np.nan)
    py = np.full(len(pts3d), np.nan)

    if in_front.sum() == 0:
        return px, py, depth, in_front

    # cv2.projectPoints wants rvec (3,1) — convert R via Rodrigues
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    tvec    = t.astype(np.float64).reshape(3, 1)

    proj, _ = cv2.projectPoints(
        pts3d[in_front].astype(np.float64),
        rvec, tvec, K, DIST
    )

    if proj is None:
        return px, py, depth, in_front

    proj = proj.reshape(-1, 2)
    px[in_front] = proj[:, 0]
    py[in_front] = proj[:, 1]

    return px, py, depth, in_front


def depth_colormap(depth: np.ndarray, max_depth: float | None = None) -> np.ndarray:
    """Map depth values to BGR colours using TURBO colormap."""
    d = depth.copy()
    positive = d[d > 0]
    if len(positive) == 0:
        return np.zeros((len(depth), 3), dtype=np.uint8)
    if max_depth:
        ceil = max_depth
    else:
        ceil = float(np.percentile(positive, 98))
    if ceil <= 0:
        ceil = positive.max()
    d = np.clip(d, 0, ceil)
    d_norm = (d / ceil * 255).astype(np.uint8)
    colors = cv2.applyColorMap(d_norm.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    return colors   # (N, 3) BGR


# ══════════════════════════════════════════════════════════════════════════════
# Output 1: LiDAR projected onto camera image
# ══════════════════════════════════════════════════════════════════════════════

def lidar_on_image(img_bgr: np.ndarray, pts3d: np.ndarray,
                   T: np.ndarray, radius: int = 2,
                   max_depth: float | None = None) -> np.ndarray:
    """
    Returns a copy of img_bgr with LiDAR points overlaid.
    Points are coloured by depth: red=close, blue=far (TURBO colormap).
    Points behind the camera or outside the image are skipped.
    Closer points are drawn on top of farther ones.
    """
    canvas = img_bgr.copy()
    h, w   = canvas.shape[:2]

    px, py, depth, in_front = project_points(pts3d, T)

    in_img = (in_front &
               np.isfinite(px) & np.isfinite(py) &
               (px >= 0) & (px < w) &
               (py >= 0) & (py < h))

    px_v  = px[in_img].astype(np.int32)
    py_v  = py[in_img].astype(np.int32)
    dep_v = depth[in_img]

    if len(dep_v) == 0:
        print('  Warning: no LiDAR points projected into image. Check transform and axis convention.')
        return canvas

    colors = depth_colormap(dep_v, max_depth)   # (N,3) BGR

    # Draw far-to-near so close points appear on top
    order = np.argsort(dep_v)[::-1]

    for i in order:
        cv2.circle(canvas, (px_v[i], py_v[i]), radius,
                   (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])), -1)

    # Depth scale legend
    _draw_depth_legend(canvas, dep_v.max() if max_depth is None else max_depth)

    pct = 100 * in_img.sum() / len(pts3d)
    cv2.putText(canvas, f'{in_img.sum():,} pts projected  ({pct:.1f}% of cloud)',
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def _draw_depth_legend(canvas: np.ndarray, max_d: float):
    """Draw a vertical depth colour bar in the top-right corner."""
    h, w   = canvas.shape[:2]
    bar_h, bar_w = 200, 18
    margin = 15
    x0, y0 = w - margin - bar_w, margin

    # Gradient bar
    for i in range(bar_h):
        t_val = i / (bar_h - 1)
        val   = int((1 - t_val) * 255)   # top=far=blue, bottom=near=red (TURBO)
        col   = cv2.applyColorMap(np.array([[val]], dtype=np.uint8),
                                  cv2.COLORMAP_TURBO)[0, 0]
        cv2.rectangle(canvas,
                      (x0, y0 + i), (x0 + bar_w, y0 + i + 1),
                      (int(col[0]), int(col[1]), int(col[2])), -1)

    cv2.rectangle(canvas, (x0, y0), (x0 + bar_w, y0 + bar_h), (180, 180, 180), 1)
    cv2.putText(canvas, f'{max_d:.1f}m', (x0 - 45, y0 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, '0.0m', (x0 - 45, y0 + bar_h + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# Output 2: image colours mapped onto the LiDAR point cloud (front view)
# ══════════════════════════════════════════════════════════════════════════════

def image_on_lidar(img_bgr: np.ndarray, pts3d: np.ndarray,
                   T: np.ndarray) -> np.ndarray:
    """
    Returns a front-view render of the LiDAR cloud where each point is
    coloured by the camera pixel it maps to.
    Points outside the image FOV are rendered in dark grey.
    """
    h_img, w_img = img_bgr.shape[:2]

    px, py, depth, in_front = project_points(pts3d, T)

    in_img = (in_front &
               np.isfinite(px) & np.isfinite(py) &
               (px >= 0) & (px < w_img) &
               (py >= 0) & (py < h_img))

    # Sample image colours for each in-image point
    point_colors = np.full((len(pts3d), 3), 40, dtype=np.uint8)  # default dark grey
    px_v = np.clip(px[in_img].astype(np.int32), 0, w_img - 1)
    py_v = np.clip(py[in_img].astype(np.int32), 0, h_img - 1)
    point_colors[in_img] = img_bgr[py_v, px_v]   # (N,3) BGR from image

    # ── Front-facing render (X right, Z up) ──────────────────────────────────
    canvas = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)

    xs, zs = pts3d[:, 0], pts3d[:, 2]
    cx, cz = xs.mean(), zs.mean()
    span   = max(np.ptp(xs), np.ptp(zs), 1.0)
    zoom   = min(RENDER_W, RENDER_H) / span * 0.85

    def world_to_canvas(wx, wz):
        cpx = (wx - cx) * zoom + RENDER_W / 2
        cpy = RENDER_H / 2 - (wz - cz) * zoom   # flip Z so up=up
        return cpx.astype(np.int32), cpy.astype(np.int32)

    # Downsample if needed
    step = max(1, len(pts3d) // MAX_POINTS)
    idx  = np.arange(0, len(pts3d), step)

    cpx, cpy = world_to_canvas(pts3d[idx, 0], pts3d[idx, 2])
    cols      = point_colors[idx]

    in_canvas = ((cpx >= 0) & (cpx < RENDER_W) & (cpy >= 0) & (cpy < RENDER_H))

    grey_mask = in_canvas & ~in_img[idx]
    canvas[cpy[grey_mask], cpx[grey_mask]] = cols[grey_mask]

    col_mask = in_canvas & in_img[idx]
    canvas[cpy[col_mask], cpx[col_mask]] = cols[col_mask]

    # Draw camera position
    cam_world = -T[:3, :3].T @ T[:3, 3]
    cam_cpx, cam_cpy = world_to_canvas(
        np.array([cam_world[0]]), np.array([cam_world[2]]))
    if 0 <= cam_cpx[0] < RENDER_W and 0 <= cam_cpy[0] < RENDER_H:
        cv2.drawMarker(canvas, (cam_cpx[0], cam_cpy[0]),
                       (255, 255, 255), cv2.MARKER_STAR, 20, 2)
        cv2.putText(canvas, 'cam', (cam_cpx[0] + 8, cam_cpy[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, 'FRONT  (X→  Z↑)  grey=outside FOV  white★=camera',
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
    pct = 100 * in_img.sum() / len(pts3d)
    cv2.putText(canvas, f'{in_img.sum():,} / {len(pts3d):,} pts in FOV  ({pct:.1f}%)',
                (10, RENDER_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (160, 160, 160), 1, cv2.LINE_AA)
    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# Directory scanner
# ══════════════════════════════════════════════════════════════════════════════

_IMAGE_EXTS      = {'.png', '.jpg', '.jpeg'}
_IMAGE_KEYWORDS  = {'image', 'cam', 'camera', 'photo', 'frame', 'rgb'}
_LIDAR_KEYWORDS  = {'lidar', 'cloud', 'scan', 'points', 'pcd'}
_XFORM_KEYWORDS  = {'transform', 'calibration', 'calib', 'extrinsic', 'T_cam'}
_XFORM_EXTS      = {'.npy', '.json'}


def _stem_matches(stem: str, keywords: set) -> bool:
    s = stem.lower()
    return any(kw in s for kw in keywords)


def find_pairs(directory: str) -> list[dict]:
    """
    Scan *directory* (non-recursively) for matched (image, lidar, transform)
    triplets.  Files are matched by shared prefix/number tokens in their stem.

    Matching strategy:
      1. Collect all candidate files per category.
      2. If exactly one file per category → treat as a single triplet.
      3. Otherwise, group by the longest common leading token (digits or
         underscore-separated words) shared across the three files.
      4. Any unmatched files are reported as warnings.

    Returns a list of dicts: {'image': path, 'lidar': path, 'transform': path}
    """
    
    # Get Recursively get all files in directory
    images     = []
    lidars     = []
    for root, _,filename in os.walk(directory):
        if filename:
            for file in filename:
                fpath = os.path.join(root, file)
                if not os.path.isfile(fpath):
                    continue
                stem, ext = os.path.splitext(file)
                ext = ext.lower()

                if ext in _IMAGE_EXTS and _stem_matches(stem, _IMAGE_KEYWORDS):
                    images.append(fpath)
                elif ext == '.npy' and _stem_matches(stem, _LIDAR_KEYWORDS):
                    lidars.append(fpath)
    # Error if either no images or lidars were found 
    if not images:
        raise FileNotFoundError(
            f'No camera image files found in "{directory}".\n'
            f'  Images must be .png/.jpg and their filename must contain one of: '
            f'{sorted(_IMAGE_KEYWORDS)}')
    if not lidars:
        raise FileNotFoundError(
            f'No LiDAR .npy files found in "{directory}".\n'
            f'  Filenames must contain one of: {sorted(_LIDAR_KEYWORDS)}')

    # ── Simple case: one of each ──────────────────────────────────────────────
    if len(images) == 1 and len(lidars) == 1:
        return [{'image': images[0], 'lidar': lidars[0]}]

    # ── Multi-file: match by shared numeric/word tokens in stem ───────────────
    def leading_token(path):
        """Extract the first numeric run or the full stem (for sorting/grouping)."""
        stem = os.path.splitext(os.path.basename(path))[0]
        # grab all digit sequences; use the first one as the group key
        import re
        nums = re.findall(r'\d+', stem)
        return nums[0].lstrip('0') or '0' if nums else stem.lower()

    img_map   = {leading_token(p): p for p in images}
    lid_map   = {leading_token(p): p for p in lidars}

    common_keys = set(img_map) & set(lid_map)

    if not common_keys:
        # Fall back: zip in sorted order with a warning
        print('  Warning: could not match files by shared token — pairing by sort order.')
        triplets = []
        for img, lid in zip(sorted(images), sorted(lidars)):
            triplets.append({'image': img, 'lidar': lid})
        return triplets

    triplets = []
    for key in sorted(common_keys, key=lambda k: k.zfill(6)):
        triplets.append({
            'image':     img_map[key],
            'lidar':     lid_map[key],
        })

    # Report any unmatched files
    unmatched = (
        [p for k, p in img_map.items() if k not in common_keys] +
        [p for k, p in lid_map.items() if k not in common_keys]
    )
    if unmatched:
        print('  Warning: the following files could not be matched to a triplet:')
        for p in unmatched:
            print(f'    {p}')

    return triplets


# ══════════════════════════════════════════════════════════════════════════════
# Single-run processing
# ══════════════════════════════════════════════════════════════════════════════

def process_one(image_path: str, lidar_path: str, transform_path: str,
                out_dir: str, radius: int, max_depth: float | None,
                no_display: bool, run_label: str = '') -> None:
    """Load files, render both outputs, save, and optionally display."""
    prefix = f'{run_label}-' if run_label else ''

    print(f'\n{prefix}Loading camera image : {image_path}')
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')
    print(f'  {img.shape[1]} × {img.shape[0]} px')

    print(f'{prefix}Loading LiDAR data   : {lidar_path}')
    pts3d, _ = load_lidar(lidar_path)
    print(f'  {len(pts3d):,} points')

    print(f'{prefix}Loading transform    : {transform_path}')
    T = load_transform(transform_path)
    #print(f'  T_cam_lidar:\n{np.round(T, 4)}')

    os.makedirs(out_dir, exist_ok=True)

    print(f'{prefix}Rendering lidar-on-image ...')
    out1  = lidar_on_image(img, pts3d, T, radius=radius, max_depth=max_depth)
    path1 = os.path.join(out_dir, prefix+'lidar_on_image.png')
    cv2.imwrite(path1, out1)
    print(f'  Saved → {path1}')

    print(f'{prefix}Rendering image-on-lidar ...')
    out2  = image_on_lidar(img, pts3d, T)
    path2 = os.path.join(out_dir, prefix+'image_on_lidar_front.png')
    cv2.imwrite(path2, out2)
    print(f'  Saved → {path2}')

    if not no_display:
        win1 = f'LiDAR on Image  {run_label}  [any key = next]'
        win2 = f'Image on LiDAR (front)  {run_label}  [any key = close]'
        cv2.namedWindow(win1, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win1, min(img.shape[1], 1400), min(img.shape[0], 900))
        cv2.imshow(win1, out1)
        print('Press any key to continue ...')
        cv2.waitKey(0)
        cv2.destroyWindow(win1)

        cv2.namedWindow(win2, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win2, RENDER_W, RENDER_H)
        cv2.imshow(win2, out2)
        print('Press any key to close ...')
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Visualise LiDAR-camera calibration (project cloud ↔ image)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    # ── Input: single-file or directory mode ──────────────────────────────────
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dir', metavar='DIR',
                        help='Directory mode: scan for matched (image, lidar, transform) triplets')
    mode.add_argument('--paths', nargs=2, default=None, metavar=('cam_path','lidar_path'),
                        help='Single-file mode, 2 inputs: camera image (.png / .jpg)  Blickfeld LiDAR .npy file')


    # ── Options ───────────────────────────────────────────────────────────────
    parser.add_argument('-t','--transform', default=None, metavar='JSON', required=True,
                        help='Required input.  JSON file with extrinsic transform:'
                                '(translation:[x,y,z], quaternion: [w,x,y,z]) '
                                'May also embed camera intrinsics '
                                '(fx, fy, cx, cy, dist_coeffs) '
                                'Otherwise, uses default values ')
    parser.add_argument('--no-display', action='store_true',
                        help='Skip interactive windows, only save images')
    parser.add_argument('--out-dir', default=None, metavar='DIR',
                        help='Root output directory '
                                '(single: default=transform dir; '
                                'batch: default=<dir>/visualizations/)')
    parser.add_argument('--max-depth', type=float, default=None,
                        help='Clip depth colour scale at this value (metres)')
    parser.add_argument('--radius', type=int, default=2,
                        help='Point radius in pixels for lidar-on-image (default: 2)')

    args = parser.parse_args()

    # ── Load intrinsics JSON (if provided) ────────────────────────────────────
    intrinsics_data = None
    # args.transform is required, including intrinsics in the json file is not 
    print(f'Checking for intrinsics in  : {args.transform}')
    try:
        intrinsics_data = load_intrinsics(args.transform)
    except Exception as e:
        print(e)
    # ══════════════════════════════════════════════════════════════════════════
    # DIRECTORY MODE
    # ══════════════════════════════════════════════════════════════════════════
    if args.dir:
        directory = os.path.abspath(args.dir)
        if not os.path.isdir(directory):
            print(f'Error: "{directory}" is not a directory.', file=sys.stderr)
            sys.exit(1)

        print(f'Scanning directory   : {directory}')
        triplets = find_pairs(directory)
        print(f'  Found {len(triplets)} matched triplet(s)\n')

        root_out = args.out_dir or os.path.join(directory, 'visualizations')

        for i, triplet in enumerate(triplets, start=1):
            label   = f'{i:03d}'

            print(f'── {label} ─────────────────────────────────────────────────')
            print(f'  image     : {triplet["image"]}')
            print(f'  lidar     : {triplet["lidar"]}')
            

            try:
                process_one(
                    image_path=triplet['image'],
                    lidar_path=triplet['lidar'],
                    transform_path=args.transform,
                    out_dir=root_out,
                    radius=args.radius,
                    max_depth=args.max_depth,
                    no_display=args.no_display,
                    run_label=label,
                )
            except Exception as exc:
                print(f'  ERROR processing {label}: {exc}')
                continue

        print(f'\nAll done. Outputs saved under: {root_out}')

    # ══════════════════════════════════════════════════════════════════════════
    # SINGLE-FILE MODE
    # ══════════════════════════════════════════════════════════════════════════
    else:
        # Validate required positional args
        missing = []
        if not args.camera_image:
            missing.append('camera_image')
        if not args.lidar_npy:
            missing.append('lidar_npy')

        # transform may come from the intrinsics JSON
        transform_path = args.transform
        if not transform_path:
            if intrinsics_data and 'translation' in intrinsics_data and 'quaternion' in intrinsics_data:
                # The intrinsics JSON itself encodes the transform — write a
                # temp path alias so process_one can load it directly
                transform_path = args.intrinsics
            else:
                missing.append('transform')

        if missing:
            parser.error(
                f'Single-file mode requires: {", ".join(missing)}.\n'
                f'  Use --dir for directory mode.')

        out_dir = args.out_dir or os.path.dirname(os.path.abspath(transform_path))

        process_one(
            image_path=args.camera_image,
            lidar_path=args.lidar_npy,
            transform_path=transform_path,
            out_dir=out_dir,
            radius=args.radius,
            max_depth=args.max_depth,
            no_display=args.no_display,
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
