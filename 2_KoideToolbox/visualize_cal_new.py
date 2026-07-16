#!/usr/bin/env python3
"""
LiDAR-Camera Calibration Visualiser
-------------------------------------
Two outputs from a saved T_cam_lidar transform:

  1. lidar_on_image  — LiDAR points projected onto the camera image,
                       coloured by depth (close=red, far=blue)

  2. image_on_lidar  — Each LiDAR point coloured by the RGB value of the
                       camera pixel it maps to (top-down view).
                       Points that fall outside the image are grey.

Usage:
  python visualize_calibration.py <camera_image> <lidar_npy> <transform_npy>

Arguments:
  camera_image   Path to camera image (.png / .jpg)
  lidar_npy      Path to Blickfeld LiDAR .npy file
  transform_npy  Path to saved 4×4 T_cam_lidar matrix (.npy)

Optional flags:
  --no-display   Skip interactive windows, only save output images
  --out-dir DIR  Directory to save output images (default: same as transform_npy)
  --max-depth M  Clip depth colour scale at M metres (default: auto)
  --radius R     Point radius in pixels for projected points (default: 2)
"""

import argparse
import os
import numpy as np
import cv2

# ── Camera intrinsics ─────────────────────────────────────────────────────────
FX   = 1248.8
FY   = 1244.6
CX   = 945.08
CY   = 527.51
DIST = np.array([0.1949, -0.3245, 0.0, 0.0, 0.0], dtype=np.float64)
K    = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)

MAX_POINTS = 500_000   # downsample above this

# ── LiDAR top-down render settings ───────────────────────────────────────────
RENDER_W = 1200
RENDER_H = 900


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
    cart[:, 0] *= -1   # negate X to match camera convention
    return cart, intensity


def load_transform(path: str) -> np.ndarray:
    T = np.load(path)
    assert T.shape == (4, 4), f'Expected 4×4 matrix, got {T.shape}'
    return T.astype(np.float64)


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
        # Nothing in front of camera — return all red as a visible error indicator
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

    # Filter to in-front + in-image
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
        t   = i / (bar_h - 1)
        val = int((1 - t) * 255)   # top=far=blue, bottom=near=red (TURBO)
        col = cv2.applyColorMap(np.array([[val]], dtype=np.uint8),
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
# Output 2: image colours mapped onto the LiDAR point cloud (top-down view)
# ══════════════════════════════════════════════════════════════════════════════

def image_on_lidar(img_bgr: np.ndarray, pts3d: np.ndarray,
                   T: np.ndarray) -> np.ndarray:
    """
    Returns a top-down render of the LiDAR cloud where each point is
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
    # This view looks along the Y axis into the scene — matches the camera FOV
    # and is most useful for indoor rooms where top-down is obscured by the ceiling.
    canvas = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)

    # Auto-fit zoom/pan using X and Z extents
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

    # Draw — grey (out of FOV) first, then coloured on top
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
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Visualise LiDAR-camera calibration (project cloud ↔ image)')
    parser.add_argument('camera_image',   help='Camera image (.png / .jpg)')
    parser.add_argument('lidar_npy',      help='Blickfeld LiDAR .npy file')
    parser.add_argument('transform_npy',  help='Saved 4×4 T_cam_lidar matrix (.npy)')
    parser.add_argument('--no-display',   action='store_true',
                        help='Skip interactive windows, only save images')
    parser.add_argument('--out-dir',      default=None,
                        help='Output directory (default: directory of transform_npy)')
    parser.add_argument('--max-depth',    type=float, default=None,
                        help='Clip depth colour scale at this value (metres)')
    parser.add_argument('--radius',       type=int, default=2,
                        help='Point radius in pixels for lidar-on-image (default: 2)')
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.transform_npy))
    os.makedirs(out_dir, exist_ok=True)

    print(f'Loading camera image : {args.camera_image}')
    img = cv2.imread(args.camera_image)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {args.camera_image}')
    print(f'  {img.shape[1]} × {img.shape[0]} px')

    print(f'Loading LiDAR data   : {args.lidar_npy}')
    pts3d, _ = load_lidar(args.lidar_npy)
    print(f'  {len(pts3d):,} points')

    print(f'Loading transform    : {args.transform_npy}')
    T = load_transform(args.transform_npy)
    T = np.linalg.inv(T)
    print(f'  T_cam_lidar:\n{np.round(T, 4)}')

    print('\nRendering lidar-on-image ...')
    out1 = lidar_on_image(img, pts3d, T,
                          radius=args.radius, max_depth=args.max_depth)
    path1 = os.path.join(out_dir, 'lidar_on_image.png')
    cv2.imwrite(path1, out1)
    print(f'  Saved → {path1}')

    print('Rendering image-on-lidar ...')
    out2 = image_on_lidar(img, pts3d, T)
    path2 = os.path.join(out_dir, 'image_on_lidar_front.png')
    cv2.imwrite(path2, out2)
    print(f'  Saved → {path2}')

    if not args.no_display:
        win1 = 'LiDAR on Image  [any key = next]'
        win2 = 'Image on LiDAR (front view)  [any key = close]'
        cv2.namedWindow(win1, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win1, min(img.shape[1], 1400), min(img.shape[0], 900))
        cv2.imshow(win1, out1)
        print('\nPress any key to continue ...')
        cv2.waitKey(0)
        cv2.destroyWindow(win1)

        cv2.namedWindow(win2, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win2, RENDER_W, RENDER_H)
        cv2.imshow(win2, out2)
        print('Press any key to close ...')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print('\nDone.')


if __name__ == '__main__':
    main()