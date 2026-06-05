#!/usr/bin/env python3
"""
Convert paired PCD + PNG files to individual ROS2 bags for direct_visual_lidar_calibration.
Creates one bag per frame pair: /image + /camera_info + /points topics.
PCD fields: x y z float32 binary. Intensity synthesized as L2 distance.
Camera intrinsics: placeholder values, update calib.json before calibrate if known.
"""
import os
import sys
import numpy as np
import cv2
import rclpy
from rclpy.serialization import serialize_message
import rosbag2_py
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from std_msgs.msg import Header
from builtin_interfaces.msg import Time

# ---- camera intrinsics (1920x1080 placeholder) ----
IMG_W, IMG_H = 1920, 1080
# FX = FY = 1200.0
# CX, CY = 960.0, 540.0
# DIST = [0.0, 0.0, 0.0, 0.0, 0.0]

FX = 1248.8
FY = 1244.6
CX = 945.08
CY = 527.51
DIST = [0.1949, -0.3245, 0.0, 0.0, 0.0]

DATA_LIDAR = "/tmp/data/lidar"
DATA_CAMERA = "/tmp/data/camera"
OUT_DIR = "/tmp/bags"

def load_blickfeld_intensity(npy_path):
    """Return float32 photon_count normalised to 0-255 from Blickfeld .npy file."""
    import blickfeld_qb2
    obj = np.load(npy_path, allow_pickle=True).item()
    ph = obj.binary.photon_count.astype(np.float32)
    p01 = np.percentile(ph, 1)
    p99 = np.percentile(ph, 99)
    ph = np.clip(ph, p01, p99)
    ph = (ph - p01) / (p99 - p01) * 255.0
    return ph.astype(np.float32)


def parse_pcd(path):
    """Return (N,3) float32 array from a binary PCD file (x y z)."""
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0]
            if key == "DATA":
                header["data_type"] = parts[1]
                break
            header[key] = parts[1:]
        raw = f.read()

    fields = header["FIELDS"]
    sizes  = [int(s) for s in header["SIZE"]]
    types  = header["TYPE"]
    counts = [int(c) for c in header["COUNT"]]
    n      = int(header["POINTS"][0])

    # Build a numpy structured dtype for the full point record
    numpy_type = {"F": {4: "f4", 8: "f8"}, "I": {4: "i4", 8: "i8"}, "U": {4: "u4", 8: "u8"}}
    dt_fields = []
    for fname, t, s, c in zip(fields, types, sizes, counts):
        base = numpy_type[t][s]
        dt_fields.append((fname, base, (c,)) if c > 1 else (fname, base))
    dt = np.dtype(dt_fields)

    pts_struct = np.frombuffer(raw[: n * dt.itemsize], dtype=dt)
    return np.column_stack([
        pts_struct["x"].astype(np.float32),
        pts_struct["y"].astype(np.float32),
        pts_struct["z"].astype(np.float32),
    ])


def make_timestamp(idx):
    t = Time()
    t.sec = idx
    t.nanosec = 0
    return t


def make_header(idx, frame_id):
    h = Header()
    h.stamp = make_timestamp(idx)
    h.frame_id = frame_id
    return h


def make_pointcloud2(pts, idx, intensity=None):
    """Build PointCloud2 with x y z intensity."""
    n = pts.shape[0]
    if intensity is None:
        intensity = np.linalg.norm(pts, axis=1).astype(np.float32)
    data = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "f4")])
    data["x"] = pts[:, 0]
    data["y"] = pts[:, 1]
    data["z"] = pts[:, 2]
    data["intensity"] = intensity

    msg = PointCloud2()
    msg.header = make_header(idx, "lidar")
    msg.height = 1
    msg.width = n
    msg.fields = [
        PointField(name="x",         offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y",         offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z",         offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * n
    msg.data = data.tobytes()
    msg.is_dense = True
    return msg


def make_image(png_path, idx):
    img_bgr = cv2.imread(png_path)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read {png_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    msg = Image()
    msg.header = make_header(idx, "camera")
    msg.height = h
    msg.width = w
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = w * 3
    msg.data = img_rgb.tobytes()
    return msg


def make_camera_info(idx):
    msg = CameraInfo()
    msg.header = make_header(idx, "camera")
    msg.width = IMG_W
    msg.height = IMG_H
    msg.distortion_model = "plumb_bob"
    msg.d = DIST
    msg.k = [FX, 0.0, CX,
             0.0, FY, CY,
             0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0]
    msg.p = [FX, 0.0, CX, 0.0,
             0.0, FY, CY, 0.0,
             0.0, 0.0, 1.0, 0.0]
    return msg


def write_bag(idx, pcd_path, png_path, out_root):
    bag_dir = os.path.join(out_root, f"frame_{idx:03d}")
    # rosbag2 creates its own directory; pre-creating it causes an error
    import shutil
    if os.path.exists(bag_dir):
        shutil.rmtree(bag_dir)

    writer = rosbag2_py.SequentialWriter()
    storage_opts = rosbag2_py.StorageOptions(uri=bag_dir, storage_id="sqlite3")
    converter_opts = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    writer.open(storage_opts, converter_opts)

    for topic, msg_type in [
        ("/image",       "sensor_msgs/msg/Image"),
        ("/camera_info", "sensor_msgs/msg/CameraInfo"),
        ("/points",      "sensor_msgs/msg/PointCloud2"),
    ]:
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=topic,
            type=msg_type,
            serialization_format="cdr",
        ))

    pts = parse_pcd(pcd_path)
    npy_path = pcd_path.replace(".pcd", ".npy")
    intensity = load_blickfeld_intensity(npy_path) if os.path.exists(npy_path) else None
    pc_msg = make_pointcloud2(pts, idx, intensity)
    img_msg = make_image(png_path, idx)
    ci_msg = make_camera_info(idx)

    ts_ns = idx * 10**9  # 1 second apart
    writer.write("/image",       serialize_message(img_msg), ts_ns)
    writer.write("/camera_info", serialize_message(ci_msg),  ts_ns)
    writer.write("/points",      serialize_message(pc_msg),  ts_ns)

    del writer
    print(f"  wrote {bag_dir}  ({pts.shape[0]} pts)")


def main():
    rclpy.init()
    os.makedirs(OUT_DIR, exist_ok=True)

    indices = sorted(
        int(f.replace("lidar_", "").replace(".pcd", ""))
        for f in os.listdir(DATA_LIDAR)
        if f.endswith(".pcd") and f.startswith("lidar_")
    )
    print(f"Found {len(indices)} frame pairs: {indices}")

    for idx in indices:
        pcd_path = os.path.join(DATA_LIDAR, f"lidar_{idx}.pcd")
        png_path = os.path.join(DATA_CAMERA, f"cam_{idx}.png")
        if not os.path.exists(png_path):
            print(f"  SKIP {idx}: missing {png_path}")
            continue
        write_bag(idx, pcd_path, png_path, OUT_DIR)

    rclpy.shutdown()
    print(f"\nDone. Bags written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
