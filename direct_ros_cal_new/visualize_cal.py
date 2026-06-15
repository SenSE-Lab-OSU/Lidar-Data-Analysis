#!/usr/bin/env python3
"""
Two visualizations to validate calibration quality:

1. projection_NNN.png  — LiDAR points projected onto camera image using T_lidar_camera.
   Points colored by range (blue=near, red=far). Misalignment shows as points
   landing on the wrong side of visible edges.

2. match_quality.png   — bar chart of per-frame match counts + confidence.

Usage:
  python3 visualize_cal.py [preprocessed_dir]
"""
import os
import sys
import json
import glob
import numpy as np
import cv2

PREPROCESS_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/preprocessed"
OUT_DIR = PREPROCESS_DIR


def load_calib(preprocess_dir):
    with open(os.path.join(preprocess_dir, "calib.json")) as f:
        d = json.load(f)
    intr = d["camera"]["intrinsics"]   # fx, fy, cx, cy
    dist = d["camera"]["distortion_coeffs"]
    res  = d["results"]
    T    = res.get("T_lidar_camera", res.get("init_T_lidar_camera_auto"))
    return intr, dist, T


def T7_to_mat(T7):
    """[tx,ty,tz,qx,qy,qz,qw] → 4×4 float64."""
    tx, ty, tz, qx, qy, qz, qw = T7
    R = np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),     2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3,  3] = [tx, ty, tz]
    return mat


def read_ply(path):
    with open(path, "rb") as f:
        n = None
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if "element vertex" in line:
                n = int(line.split()[-1])
            if line == "end_header":
                break
        raw = f.read()
    dt = np.dtype([("x","f4"),("y","f4"),("z","f4"),("intensity","f4")])
    return np.frombuffer(raw[:n*dt.itemsize], dtype=dt)


def project_points(pts_xyz, T_lidar_cam, intr, dist, img_shape):
    """Return (u, v, depth) arrays for points projecting inside the image."""
    fx, fy, cx, cy = intr
    h, w = img_shape[:2]

    # T_lidar_camera transforms from camera frame to lidar frame.
    # We need T_camera_lidar = inv(T_lidar_camera) to put points in camera frame.
    T_cam_lid = np.linalg.inv(T_lidar_cam)
    R = T_cam_lid[:3, :3]
    t = T_cam_lid[:3, 3]

    pts_cam = (R @ pts_xyz.T).T + t          # (N,3) in camera frame
    z = pts_cam[:, 2]
    front = z > 0.1
    pts_cam = pts_cam[front]
    z = z[front]
    if len(z) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([])

    # project with distortion via OpenCV
    k_mat = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    d_vec = np.array(dist, dtype=np.float64)
    pts_cam_cv = pts_cam[:, :3].astype(np.float64)
    uv, _ = cv2.projectPoints(
        pts_cam_cv.reshape(-1,1,3), np.zeros(3), np.zeros(3), k_mat, d_vec
    )
    uv = uv.reshape(-1, 2)
    u, v = uv[:, 0], uv[:, 1]

    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u[inside].astype(int), v[inside].astype(int), z[inside]


def colorize_depth(depth):
    """Map depths to BGR colors using COLORMAP_JET (blue=near, red=far)."""
    d = np.array(depth, dtype=np.float32)
    p2, p98 = np.percentile(d, 2), np.percentile(d, 98)
    norm = np.clip((d - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)
    colors = cv2.applyColorMap(norm.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
    return colors


def make_projection_image(cam_png, pts, T_lidar_cam, intr, dist, dot_size=3):
    img = cv2.imread(cam_png, cv2.IMREAD_COLOR)
    if img is None:
        return None
    xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float64)
    u, v, depth = project_points(xyz, T_lidar_cam, intr, dist, img.shape)
    if len(u) == 0:
        return img
    colors = colorize_depth(depth)
    for i in range(len(u)):
        cv2.circle(img, (u[i], v[i]), dot_size, colors[i].tolist(), -1)
    return img


def make_match_quality_chart(preprocess_dir):
    """Bar chart: match count and high-conf count per frame."""
    records = []
    for mf in sorted(glob.glob(os.path.join(preprocess_dir, "*_matches.json"))):
        m     = json.load(open(mf))
        mch   = np.array(m["matches"])
        conf  = np.array(m["confidence"])
        valid = mch >= 0
        n     = int(valid.sum())
        hi    = int((conf[valid] > 0.5).sum())
        name  = os.path.basename(mf).replace("_matches.json", "")
        records.append((name, n, hi))
    if not records:
        return None

    n_frames = len(records)
    bar_w, bar_h = 40, 300
    gap, left_margin, bottom_margin = 4, 60, 40
    total_w = left_margin + n_frames * (bar_w + gap) + 20
    total_h = bar_h + bottom_margin + 60
    img = np.ones((total_h, total_w, 3), dtype=np.uint8) * 240

    max_n = max(r[1] for r in records)
    scale = bar_h / max(max_n, 1)

    for i, (name, n, hi) in enumerate(records):
        x = left_margin + i * (bar_w + gap)
        # total bar (light blue)
        h_total = int(n * scale)
        cv2.rectangle(img,
                       (x, bar_h + bottom_margin - h_total),
                       (x + bar_w, bar_h + bottom_margin),
                       (200, 150, 80), -1)
        # high-conf bar (green)
        h_hi = int(hi * scale)
        cv2.rectangle(img,
                       (x, bar_h + bottom_margin - h_hi),
                       (x + bar_w, bar_h + bottom_margin),
                       (60, 180, 60), -1)
        # frame label (rotated)
        label = name.replace("frame_", "")
        cv2.putText(img, label, (x + 2, bar_h + bottom_margin + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 50), 1)

    # Y-axis labels
    for tick in [0, 50, 100, 150, 200]:
        y = bar_h + bottom_margin - int(tick * scale)
        if y >= bottom_margin and y <= bar_h + bottom_margin:
            cv2.line(img, (left_margin - 5, y), (left_margin, y), (100, 100, 100), 1)
            cv2.putText(img, str(tick), (2, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 50), 1)

    # legend
    cv2.rectangle(img, (left_margin, 5), (left_margin+14, 19), (200, 150, 80), -1)
    cv2.putText(img, "all matches", (left_margin+18, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50,50,50), 1)
    cv2.rectangle(img, (left_margin+110, 5), (left_margin+124, 19), (60, 180, 60), -1)
    cv2.putText(img, "conf>0.5", (left_margin+128, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50,50,50), 1)

    return img


def main():
    intr, dist, T7 = load_calib(PREPROCESS_DIR)
    T_lidar_cam = T7_to_mat(T7)
    print(f"Using T_lidar_camera: t=[{T7[0]:.3f}, {T7[1]:.3f}, {T7[2]:.3f}]")
    print(f"  intrinsics: fx={intr[0]:.1f} fy={intr[1]:.1f} cx={intr[2]:.1f} cy={intr[3]:.1f}")

    # 1. Projection overlays
    ply_files = sorted(glob.glob(os.path.join(PREPROCESS_DIR, "*.ply")))
    for ply_path in ply_files:
        base = os.path.splitext(ply_path)[0]
        cam_png = base + ".png"
        out_png = base + "_projection.png"
        if not os.path.exists(cam_png):
            continue
        pts = read_ply(ply_path)
        img = make_projection_image(cam_png, pts, T_lidar_cam, intr, dist)
        if img is not None:
            cv2.imwrite(out_png, img)
            print(f"wrote {os.path.basename(out_png)}")

    # 2. Match quality chart
    chart = make_match_quality_chart(PREPROCESS_DIR)
    if chart is not None:
        chart_path = os.path.join(OUT_DIR, "match_quality.png")
        cv2.imwrite(chart_path, chart)
        print(f"wrote match_quality.png")


if __name__ == "__main__":
    main()
