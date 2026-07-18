#!/usr/bin/env python3
"""
read_cam_lidar.py

Synchronously captures frames from a USB camera and a Blickfeld QB2 LiDAR,
saving matched pairs to disk for later processing.

Output layout (under --save-dir, default: current working directory):

    <save-dir>/
        camera/
            cam_<N>.png
        lidar/
            lidar_<N>.npy   # raw Blickfeld frame object 
            lidar_<N>.pcd   # point cloud in PCD format for Koide create_bags.py

Frame numbering resumes automatically from the highest existing index found
in the output directories, so re-running the script on an existing save
directory will not overwrite previous captures.

Usage:
    python read_cam_lidar.py [--save-dir DIR] [--lidar-ip IP] [--camera-index N]
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
from pypcd4 import PointCloud

import os

# Significant speed improvement when set BEFORE importing cv2.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
import cv2 as cv  # noqa: E402

import blickfeld_qb2  # noqa: E402

LOGGER = logging.getLogger("read_cam_lidar")

DEFAULT_LIDAR_IP = "192.168.0.253"
DEFAULT_CAMERA_INDEX = 1
DEFAULT_CAMERA_WIDTH = 3840
DEFAULT_CAMERA_HEIGHT = 2160
DEFAULT_CAMERA_ZOOM = None  # I am using whatever value zoom is default at, set this var to something if you know better!!
DEFAULT_CAMERA_FOCUS = None  # I am using whatever value zoom is default at, set this var to something if you know better!!

LIDAR_DIRNAME = "lidar"
CAMERA_DIRNAME = "camera"

LIDAR_FILE_PATTERN = re.compile(r"lidar_(\d+)\.npy$")
CAMERA_FILE_PATTERN = re.compile(r"cam_(\d+)\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture synchronized camera + Blickfeld LiDAR frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path.cwd(),
        help="Root directory to save 'camera/' and 'lidar/' subfolders into. "
        "Defaults to the current working directory.",
    )
    parser.add_argument(
        "--lidar-ip",
        default=DEFAULT_LIDAR_IP,
        help="IP address (or FQDN) of the Blickfeld QB2 LiDAR.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help="OpenCV camera index to open (0 is usually the default/built-in camera).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


def get_next_save_index(lidar_dir: Path, camera_dir: Path) -> int:
    """Inspect existing saved files and return the next frame index to use.

    Also warns (but does not fail) if the lidar and camera directories are
    out of sync, e.g. due to a previous crashed/interrupted run.
    """
    lidar_numbers = sorted(
        int(m.group(1))
        for f in lidar_dir.iterdir()
        if f.is_file() and (m := LIDAR_FILE_PATTERN.match(f.name))
    )
    camera_numbers = sorted(
        int(m.group(1))
        for f in camera_dir.iterdir()
        if f.is_file() and (m := CAMERA_FILE_PATTERN.match(f.name))
    )

    if not lidar_numbers and not camera_numbers:
        return 0

    if lidar_numbers != camera_numbers:
        LOGGER.warning(
            "Lidar and camera directories are out of sync (%d lidar files vs "
            "%d camera files). Mismatched indices: %s",
            len(lidar_numbers),
            len(camera_numbers),
            sorted(set(lidar_numbers) ^ set(camera_numbers)),
        )

    return max(lidar_numbers + camera_numbers) + 1


def open_camera(camera_index: int, width: int = DEFAULT_CAMERA_WIDTH, height: int = DEFAULT_CAMERA_HEIGHT) -> cv.VideoCapture:
    LOGGER.info("Opening camera at index %d ...", camera_index)
    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {camera_index}")

    # Disable autofocus so focus stays fixed/consistent across all captured
    # frames. Not all backends support this, so we verify it took effect.
    cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
    # I don't know what default values are so I am setting them to whatever they are at runtime--Change maybe!!
    if DEFAULT_CAMERA_ZOOM is not None:
        cap.set(cv.CAP_PROP_ZOOM,zoom)
    else:
        zoom = cap.get(cv.CAP_PROP_ZOOM)
        cap.set(cv.CAP_PROP_ZOOM,zoom)
    if DEFAULT_CAMERA_FOCUS is not None:
        cap.set(cv.CAP_PROP_FOCUS,focus)
    else:
        focus = cap.get(cv.CAP_PROP_FOCUS)
        cap.set(cv.CAP_PROP_FOCUS,focus)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT,height)
    reported_w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    reported_h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    autofocus_state = cap.get(cv.CAP_PROP_AUTOFOCUS)

    if autofocus_state != 0:
        LOGGER.warning(
            "Autofocus may still be enabled (CAP_PROP_AUTOFOCUS reports %s); "
            "this camera/backend may not support disabling it via OpenCV.",
            autofocus_state,
        )
    else:
        LOGGER.info("Autofocus disabled.")

    ret, test_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Camera opened but failed to read a test frame")

    # The actual captured frame shape is the ground truth for resolution;
    # cap.get() values can be inaccurate/rounded depending on the backend.
    actual_h, actual_w = test_frame.shape[:2]
    LOGGER.info(
        "Camera ready. Requested width=%d. Driver-reported size=%dx%d. "
        "Actual captured frame size=%dx%d (width x height) -- all saved "
        "images will be at this resolution.",
        width,
        reported_w,
        reported_h,
        actual_w,
        actual_h,
    )

    return cap


def capture_loop(
    cap: cv.VideoCapture,
    lidar_ip: str,
    lidar_dir: Path,
    camera_dir: Path,
    start_index: int,
) -> None:
    """Interactively capture matched lidar/camera frame pairs until the user quits."""
    frame_id = start_index

    with blickfeld_qb2.Channel(fqdn_or_ip=lidar_ip) as channel:
        service = blickfeld_qb2.core_processing.services.PointCloud(channel)

        while True:
            lidar_frame = service.get().frame
            point_cloud = PointCloud.from_xyz_points(lidar_frame.binary.cartesian)
            cam_ok, cam_frame = cap.read()

            if not cam_ok or not lidar_frame.id:
                LOGGER.error("Failed to receive a valid lidar and/or camera frame; retrying.")
                continue

            lidar_npy_path = lidar_dir / f"lidar_{frame_id}.npy"
            lidar_pcd_path = lidar_dir / f"lidar_{frame_id}.pcd"
            camera_png_path = camera_dir / f"cam_{frame_id}.png"

            np.save(lidar_npy_path, lidar_frame)
            point_cloud.save(str(lidar_pcd_path))
            cv.imwrite(str(camera_png_path), cam_frame)

            LOGGER.info("Saved frame pair %d:", frame_id)
            LOGGER.info("  %s", lidar_npy_path)
            LOGGER.info("  %s", lidar_pcd_path)
            LOGGER.info("  %s", camera_png_path)

            user_in = input("Press 't' to terminate, or Enter to capture the next frame: ").strip().lower()
            if user_in == "t":
                break

            frame_id += 1


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    save_dir: Path = args.save_dir
    lidar_dir = save_dir / LIDAR_DIRNAME
    camera_dir = save_dir / CAMERA_DIRNAME

    if not save_dir.exists():
        LOGGER.info("Save directory %s does not exist, creating it.", save_dir)
    lidar_dir.mkdir(parents=True, exist_ok=True)
    camera_dir.mkdir(parents=True, exist_ok=True)

    start_index = get_next_save_index(lidar_dir, camera_dir)
    LOGGER.info(
        "First save will be:\n\t%s\n\t%s\n\t%s",
        lidar_dir / f"lidar_{start_index}.npy",
        lidar_dir / f"lidar_{start_index}.pcd",
        camera_dir / f"cam_{start_index}.png",
    )

    cap = open_camera(args.camera_index)
    try:
        capture_loop(cap, args.lidar_ip, lidar_dir, camera_dir, start_index)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user, shutting down.")
    finally:
        cap.release()
        LOGGER.info("Camera released. Done.")


if __name__ == "__main__":
    main()