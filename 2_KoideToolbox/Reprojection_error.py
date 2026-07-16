#!/usr/bin/env python3
"""
reprojection_error.py

Computes reprojection error for a single calibrated frame from Koide's
direct_visual_lidar_calibration toolbox.

Pipeline:
  1. Read calib.json  -> camera intrinsics/distortion + T_lidar_camera (LiDAR<-Camera)
  2. Read frame_###_matches.json -> 2D camera keypoints <-> 2D synthetic-lidar-image keypoints
  3. Read frame_###_lidar_indices.png -> for each lidar-image pixel, the index of the
     3D point (in the corresponding .ply) that produced it
  4. Read frame_###.ply -> the actual 3D points, in the LiDAR frame
  5. Transform each matched 3D point into the camera frame (using inverse of
     T_lidar_camera) and project it with the camera model + distortion
  6. Compare the projected pixel to the matched camera-image keypoint

-----------------------------------------------------------------------------
TWO THINGS IN THIS SCRIPT ARE ASSUMPTIONS, NOT CONFIRMED FROM SOURCE:

  (A) The key names inside frame_###_matches.json.
      -> See CAMERA_KP_KEYS / LIDAR_KP_KEYS / MATCH_IDX_KEYS / CONF_KEYS below.
  (B) How frame_###_lidar_indices.png packs a point-cloud index into pixel bytes.
      -> See decode_index_map().

Run with --inspect first on your actual files. It will print the real keys
and dtypes so you can fix the two spots above if my guess doesn't match.
-----------------------------------------------------------------------------
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2

try:
    from plyfile import PlyData
except ImportError:
    PlyData = None


# ---------------------------------------------------------------------------
# (A) Adjust these if --inspect shows different key names in your matches.json
# ---------------------------------------------------------------------------
CAMERA_KP_KEYS = ["kpts0"]
LIDAR_KP_KEYS = ["kpts1"]
MATCH_IDX_KEYS = ["matches", "match_indices"]
CONF_KEYS = ["match_confidence", "confidence", "matching_scores0"]


# ---------------------------------------------------------------------------
# calib.json
# ---------------------------------------------------------------------------
def load_calib(path):
    with open(path) as f:
        calib = json.load(f)

    cam = calib["camera"]
    fx, fy, cx, cy = cam["intrinsics"]
    dist = np.array(cam.get("distortion_coeffs", []), dtype=np.float64)
    camera_model = cam.get("camera_model", "plumb_bob")

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float64)

    tlc = calib["results"]["T_lidar_camera"]  # [x,y,z,qx,qy,qz,qw], camera->lidar
    T_lidar_camera = quat_xyzw_to_matrix(tlc[:3], tlc[3:])
    T_camera_lidar = invert_transform(T_lidar_camera)

    return {
        "K": K,
        "dist": dist,
        "camera_model": camera_model,
        "T_lidar_camera": T_lidar_camera,
        "T_camera_lidar": T_camera_lidar,
    }


def quat_xyzw_to_matrix(t, q):
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat(q).as_matrix()  # q = [qx,qy,qz,qw]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


# ---------------------------------------------------------------------------
# matches.json
# ---------------------------------------------------------------------------
def _first_present(d, keys):
    for k in keys:
        if k in d:
            return k
    return None


def load_matches(path):
    with open(path) as f:
        data = json.load(f)

    cam_key = _first_present(data, CAMERA_KP_KEYS)
    lidar_key = _first_present(data, LIDAR_KP_KEYS)
    if cam_key is None or lidar_key is None:
        raise KeyError(
            f"Could not find camera/lidar keypoint keys in {path}. "
            f"Top-level keys are: {list(data.keys())}. "
            f"Update CAMERA_KP_KEYS / LIDAR_KP_KEYS at the top of this script."
        )

    cam_kpts = np.asarray(data[cam_key], dtype=np.float64).reshape(-1, 2)
    lidar_kpts = np.asarray(data[lidar_key], dtype=np.float64).reshape(-1, 2)

    match_key = _first_present(data, MATCH_IDX_KEYS)
    conf_key = _first_present(data, CONF_KEYS)
    conf = np.array(data[conf_key], dtype=np.float64) if conf_key else None

    if match_key is not None:
        # Full keypoint sets + separate match index array (SuperGlue match_pairs.py style):
        # matches[i] == j means cam_kpts[i] <-> lidar_kpts[j], -1 = unmatched
        match_idx = np.array(data[match_key], dtype=np.int64)
        valid = match_idx > -1
        pairs = np.stack([cam_kpts[valid], lidar_kpts[match_idx[valid]]], axis=1)
        pair_conf = conf[valid] if conf is not None else None
    else:
        # Already-paired, equal-length arrays
        if len(cam_kpts) != len(lidar_kpts):
            raise ValueError(
                f"cam_kpts ({len(cam_kpts)}) and lidar_kpts ({len(lidar_kpts)}) "
                f"have different lengths and no match-index array was found."
            )
        pairs = np.stack([cam_kpts, lidar_kpts], axis=1)
        pair_conf = conf

    return pairs, pair_conf  # pairs: (N, 2, 2) -> [i, 0]=cam(u,v)  [i, 1]=lidar(u,v)


# ---------------------------------------------------------------------------
# (B) lidar_indices.png decoding
# ---------------------------------------------------------------------------
def decode_index_map(png_path, invalid_index=None):
    img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(png_path)

    if img.ndim == 2:
        # single channel -> assume it already IS the index (uint16 or uint32-as-int32)
        idx = img.astype(np.int64)
        default_invalid = int(np.iinfo(img.dtype).max)
    elif img.ndim == 3:
        # cv2 loads as BGR[A]; assume little-endian byte packing:
        # index = B + G*256 + R*65536 (+ A*16777216)
        b = img[..., 0].astype(np.int64)
        g = img[..., 1].astype(np.int64)
        r = img[..., 2].astype(np.int64)
        idx = b + (g << 8) + (r << 16)
        if img.shape[2] == 4:
            a = img[..., 3].astype(np.int64)
            idx = idx + (a << 24)
            default_invalid = 0xFFFFFFFF
        else:
            default_invalid = 0xFFFFFF
    else:
        raise ValueError(f"Unexpected index map shape: {img.shape}")

    inv = invalid_index if invalid_index is not None else default_invalid
    valid_mask = idx != inv
    return idx, valid_mask


def load_points(ply_path):
    if PlyData is None:
        raise ImportError("pip install plyfile --break-system-packages")
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    return pts


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
def project_points(pts_cam, K, dist, camera_model):
    """pts_cam: (N,3) points already in the camera frame."""
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    if camera_model == "fisheye":
        d = np.zeros(4) if dist.size == 0 else dist[:4]
        proj, _ = cv2.fisheye.projectPoints(
            pts_cam.reshape(-1, 1, 3), rvec, tvec, K, d
        )
    else:
        # plumb_bob / standard OpenCV radial-tangential model
        proj, _ = cv2.projectPoints(pts_cam, rvec, tvec, K, dist)
    return proj.reshape(-1, 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute reprojection error for all frames in a calibration directory")

    ap.add_argument(
        "data_path",
        help="Directory containing calib.json and all frame files"
    )
    ap.add_argument(
        "--invalid-index",
        type=int,
        default=None,
        help="Override sentinel value meaning 'no lidar return'"
    )
    ap.add_argument(
        "--csv",
        action="store_true",
        help="Write one CSV file per frame"
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Show histogram of all reprojection errors"
    )
    ap.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect first frame and exit"
    )
    ap.add_argument(
        "--silent",
        action="store_true",
        help="Prevents printing results"
    )

    args = ap.parse_args()

    data_path = Path(args.data_path)

    calib_file = data_path / "calib.json"

    if not calib_file.exists():
        raise FileNotFoundError(calib_file)

    with open(calib_file) as f:
        calib_json = json.load(f)

    calib = load_calib(calib_file)

    if "meta" not in calib_json or "bag_names" not in calib_json["meta"]:
        raise RuntimeError(
            "Could not find meta/bag_names in calib.json"
        )

    bag_names = calib_json["meta"]["bag_names"]

    if len(bag_names) == 0:
        raise RuntimeError("No bag_names found in calib.json")

    # ------------------------------------------------------------------
    # Inspect mode
    # ------------------------------------------------------------------
    if args.inspect:
        bag = bag_names[0]

        matches_file = data_path / f"{bag}_matches.json"
        indices_file = data_path / f"{bag}_lidar_indices.png"

        print(f"Inspecting frame: {bag}")

        with open(matches_file) as f:
            data = json.load(f)

        print(f"\nmatches.json keys:")
        print(list(data.keys()))

        for k, v in data.items():
            if isinstance(v, list):
                print(f"  {k}: len={len(v)}")

        idx_img = cv2.imread(str(indices_file), cv2.IMREAD_UNCHANGED)

        print(
            f"\nindices png: shape={idx_img.shape}, "
            f"dtype={idx_img.dtype}, "
            f"min={idx_img.min()}, "
            f"max={idx_img.max()}"
        )

        return

    all_errors = []
    total_matches = 0
    total_used = 0
    overall_results = []

    print(f"Found {len(bag_names)} frames")

    for bag in bag_names:

        matches_file = data_path / f"{bag}_matches.json"
        indices_file = data_path / f"{bag}_lidar_indices.png"
        ply_file = data_path / f"{bag}.ply"

        if not args.silent:
            print("\n" + "=" * 80)
            print(f"Processing {bag}")
            print("=" * 80)

        if not matches_file.exists():
            print(f"Skipping: missing {matches_file.name}")
            continue

        if not indices_file.exists():
            print(f"Skipping: missing {indices_file.name}")
            continue

        if not ply_file.exists():
            print(f"Skipping: missing {ply_file.name}")
            continue

        pairs, conf = load_matches(matches_file)
        idx_map, valid_mask = decode_index_map(
            indices_file,
            args.invalid_index
        )
        points_lidar = load_points(ply_file)

        h, w = idx_map.shape[:2]

        errors = []
        used = []
        used_conf = []

        for i, (cam_uv, lidar_uv) in enumerate(pairs):

            u = int(round(lidar_uv[0]))
            v = int(round(lidar_uv[1]))

            if not (0 <= u < w and 0 <= v < h):
                continue

            if not valid_mask[v, u]:
                continue

            pt_idx = idx_map[v, u]

            if pt_idx < 0 or pt_idx >= len(points_lidar):
                continue

            p_lidar = points_lidar[pt_idx]

            p_cam = (
                calib["T_camera_lidar"]
                @ np.append(p_lidar, 1.0)
            )[:3]

            if p_cam[2] <= 0:
                continue

            proj_uv = project_points(
                p_cam.reshape(1, 3),
                calib["K"],
                calib["dist"],
                calib["camera_model"]
            )[0]

            err = np.linalg.norm(proj_uv - cam_uv)

            confidence = conf[i] if conf is not None else np.nan

            errors.append(err)
            used.append(i)
            used_conf.append(confidence)

            overall_results.append({
                "frame": bag,
                "match_idx": i,
                "cam_u": cam_uv[0],
                "cam_v": cam_uv[1],
                "lidar_u": lidar_uv[0],
                "lidar_v": lidar_uv[1],
                "confidence": confidence,
                "error_px": err,
                "error_confidence": err * confidence,
                "quality": confidence / (1+err)
            })

        errors = np.asarray(errors)

        n_total = len(pairs)
        n_used = len(errors)

        total_matches += n_total
        total_used += n_used

        if n_used == 0:
            print("No usable matches")
            continue

        all_errors.extend(errors.tolist())

        if not args.silent:
            print(f"Matches total:        {n_total}")
            print(f"Matches used:         {n_used}")
            print(f"Mean error:           {errors.mean():.3f} px")
            print(f"Median error:         {np.median(errors):.3f} px")
            print(f"RMSE:                 {np.sqrt((errors ** 2).mean()):.3f} px")
            print(f"Max error:            {errors.max():.3f} px")
            print(f"Std dev:              {errors.std():.3f} px")

        if args.csv:
            import csv

            csv_file = data_path / f"{bag}_reprojection_error.csv"

            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)

                writer.writerow(
                    [
                        "match_idx",
                        "cam_u",
                        "cam_v",
                        "confidence",
                        "error_px",
                        "Err_conf_prod"
                    ]
                )

                for idx, err, c in zip(used, errors, used_conf):
                    writer.writerow(
                        [
                            idx,
                            pairs[idx, 0, 0],
                            pairs[idx, 0, 1],
                            c,
                            err,
                            c*err
                        ]
                    )

            print(f"Wrote {csv_file.name}")

    if len(all_errors) == 0:
        print("\nNo valid reprojection errors computed.")
        sys.exit(1)

    all_errors = np.asarray(all_errors)

    print("\n")
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    print(f"Frames processed:      {len(bag_names)}")
    print(f"Total matches:         {total_matches}")
    print(f"Matches used:          {total_used}")

    print(f"Mean error:            {all_errors.mean():.3f} px")
    print(f"Median error:          {np.median(all_errors):.3f} px")
    print(f"RMSE:                  {np.sqrt((all_errors ** 2).mean()):.3f} px")
    print(f"Max error:             {all_errors.max():.3f} px")
    print(f"Std dev:               {all_errors.std():.3f} px")

    if args.csv:
        import csv

        overall_csv = data_path / "overall_reprojection_error.csv"

        with open(overall_csv, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "frame",
                "match_idx",
                "cam_u",
                "cam_v",
                "lidar_u",
                "lidar_v",
                "confidence",
                "error_px",
                "error_confidence",
                "quality"
            ])

            for row in overall_results:
                writer.writerow([
                    row["frame"],
                    row["match_idx"],
                    row["cam_u"],
                    row["cam_v"],
                    row["lidar_u"],
                    row["lidar_v"],
                    row["confidence"],
                    row["error_px"],
                    row["error_confidence"],
                    row['quality']
                ])

        print(f"Wrote {overall_csv.name}")


    if args.plot:
        import matplotlib.pyplot as plt

        #plt.hist(all_errors, bins=40)
        plt.scatter(
            used_conf,
            errors,
            c=used_conf,
            cmap="viridis",
            s=6
        )

        plt.colorbar(label="Confidence")
        
        plt.xlabel("Reprojection error (px)")
        plt.ylabel("Count")
        plt.title("All reprojection errors")
        plt.show()

if __name__ == "__main__":
    main()  