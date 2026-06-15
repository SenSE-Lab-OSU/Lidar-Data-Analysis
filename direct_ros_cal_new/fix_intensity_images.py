#!/usr/bin/env python3
"""
Regenerate *_lidar_intensities.png for direct_visual_lidar_calibration.

Problem: the preprocess tool creates a full-sphere equirectangular image but the
Blickfeld QB2 only covers 90°x49°, so only ~5% of the 1024x1024 image is filled.

Fix: project onto a canvas that spans exactly the LiDAR FoV (linear az/el mapping),
then fill the Lissajous-pattern gaps with morphological dilation before saving.
The result is a ~70-80% filled image instead of 5%, which SuperGlue can match.
"""
import os
import sys
import glob
import numpy as np
import cv2

PREPROCESS_DIR = str(sys.argv[1]) if len(sys.argv) > 1 else "/tmp/preprocessed"
OUT_W = 1024
OUT_H = 1024

# Slightly pad the angular bounds so edge points are not cropped
AZ_PAD_DEG = 1.0
EL_PAD_DEG = 1.0

# Blend weight: 0.0 = pure reflectivity, 1.0 = pure range.
# Range has strong edges at depth discontinuities (aligns with camera edges).
# Reflectivity captures material differences. 0.5 blends both equally.
RANGE_WEIGHT = float(sys.argv[2])/100 if len(sys.argv) >  2 else 0.5


def read_ply_xyz_intensity(path):
    with open(path, "rb") as f:
        n = None
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if "element vertex" in line:
                n = int(line.split()[-1])
            if line == "end_header":
                break
        raw = f.read()
    dt = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "f4")])
    return np.frombuffer(raw[: n * dt.itemsize], dtype=dt)


def fov_matched_projection(pts, w, h, az_bounds=None, el_bounds=None):
    """
    Linear azimuth/elevation → pixel mapping covering exactly the LiDAR FoV.
    Returns (h, w) float32 image and (h, w) bool hit mask.
    """
    x, y, z = pts["x"], pts["y"], pts["z"]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    valid = r > 1e-6
    az = np.where(valid, np.degrees(np.arctan2(y, x)), 0.0)
    el = np.where(valid, np.degrees(np.arcsin(np.clip(z / np.where(valid, r, 1.0), -1, 1))), 0.0)

    if az_bounds is None:
        az_min, az_max = az[valid].min() - AZ_PAD_DEG, az[valid].max() + AZ_PAD_DEG
    else:
        az_min, az_max = az_bounds

    if el_bounds is None:
        el_min, el_max = el[valid].min() - EL_PAD_DEG, el[valid].max() + EL_PAD_DEG
    else:
        el_min, el_max = el_bounds

    az_span = az_max - az_min
    el_span = el_max - el_min

    # az_max - az so that physical-right (low az) maps to right pixel, matching camera convention
    px = np.clip(((az_max - az) / az_span * (w - 1)).astype(np.int32), 0, w - 1)
    py = np.clip(((el_max - el) / el_span * (h - 1)).astype(np.int32), 0, h - 1)

    refl_sum = np.zeros((h, w), dtype=np.float64)
    range_sum = np.zeros((h, w), dtype=np.float64)
    img_cnt   = np.zeros((h, w), dtype=np.int32)
    refl  = pts["intensity"].astype(np.float64)
    np.add.at(refl_sum,  (py, px), refl)
    np.add.at(range_sum, (py, px), r.astype(np.float64))
    np.add.at(img_cnt,   (py, px), 1)

    hit = img_cnt > 0
    refl_avg  = np.zeros((h, w), dtype=np.float32)
    range_avg = np.zeros((h, w), dtype=np.float32)
    refl_avg[hit]  = (refl_sum[hit]  / img_cnt[hit]).astype(np.float32)
    range_avg[hit] = (range_sum[hit] / img_cnt[hit]).astype(np.float32)

    # Independently normalize each channel over hit pixels to [0,1], then blend
    def norm01(arr, mask):
        out = np.zeros_like(arr)
        if mask.sum() == 0:
            return out
        vals = arr[mask]
        lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
        if hi <= lo:
            hi = lo + 1e-6
        out[mask] = np.clip((arr[mask] - lo) / (hi - lo), 0.0, 1.0)
        return out

    img = (RANGE_WEIGHT * norm01(range_avg, hit) +
           (1.0 - RANGE_WEIGHT) * norm01(refl_avg, hit)).astype(np.float32)
    return img, hit, (az_min, az_max, el_min, el_max)


def render_intensity(img_float, hit, dilation_iters=3):
    """Scale blended [0,1] image to uint8; fill Lissajous gaps with dilation."""
    out = np.clip(img_float * 255, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.dilate(out, kernel, iterations=dilation_iters)
    return out


def make_indices_image(pts, px, py, w, h):
    """
    Build the _lidar_indices.png that initial_guess_auto needs:
    each pixel stores the PLY point index as a 32-bit int packed in 4 RGBA uint8 bytes.
    -1 (0xFFFFFFFF) means no point.
    When multiple points hit the same pixel the last one wins (all are valid for bearing estimation).
    """
    indices = np.full((h, w), -1, dtype=np.int32)
    # iterate in reverse so index 0 has priority at collisions (arbitrary but consistent)
    for i in range(len(px) - 1, -1, -1):
        indices[py[i], px[i]] = i
    # reinterpret int32 as 4×uint8 (RGBA)
    return indices.view(np.uint8).reshape(h, w, 4)


def main():
    ply_files = sorted(glob.glob(os.path.join(PREPROCESS_DIR, "*.ply")))
    if not ply_files:
        print("No PLY files found in", PREPROCESS_DIR)
        sys.exit(1)

    # Compute global az/el bounds from the first file so all frames share the same projection
    pts0 = read_ply_xyz_intensity(ply_files[0])
    _, _, bounds = fov_matched_projection(pts0, OUT_W, OUT_H)
    az_min, az_max, el_min, el_max = bounds
    print(f"FoV bounds: az [{az_min:.1f}, {az_max:.1f}]°  el [{el_min:.1f}, {el_max:.1f}]°")

    for ply_path in ply_files:
        base = os.path.splitext(ply_path)[0]
        intensity_out = base + "_lidar_intensities.png"
        indices_out   = base + "_lidar_indices.png"

        pts = read_ply_xyz_intensity(ply_path)

        # Recompute per-point pixel positions (needed for both images)
        x, y, z = pts["x"], pts["y"], pts["z"]
        r = np.sqrt(x**2 + y**2 + z**2)
        valid = r > 1e-6
        az = np.where(valid, np.degrees(np.arctan2(y, x)), 0.0)
        el = np.where(valid, np.degrees(np.arcsin(np.clip(z / np.where(valid, r, 1.0), -1, 1))), 0.0)

        az_span = az_max - az_min
        el_span = el_max - el_min
        px = np.clip(((az_max - az) / az_span * (OUT_W - 1)).astype(np.int32), 0, OUT_W - 1)
        py = np.clip(((el_max - el) / el_span * (OUT_H - 1)).astype(np.int32), 0, OUT_H - 1)

        # Intensity image
        img_float, hit, _ = fov_matched_projection(
            pts, OUT_W, OUT_H,
            az_bounds=(az_min, az_max),
            el_bounds=(el_min, el_max),
        )
        out = render_intensity(img_float, hit)
        cv2.imwrite(intensity_out, out)

        # Indices image
        idx_img = make_indices_image(pts, px, py, OUT_W, OUT_H)
        cv2.imwrite(indices_out, idx_img)

        nz = (out > 0).sum()
        valid_idx = (idx_img.view(np.int32).reshape(OUT_H, OUT_W) >= 0).sum()
        print(f"{os.path.basename(ply_path)}: intensity {100*nz/out.size:.1f}% filled  "
              f"std={out.std():.1f}  indices valid={valid_idx}")


if __name__ == "__main__":
    main()
