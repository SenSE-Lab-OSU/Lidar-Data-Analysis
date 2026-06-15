# LiDAR-Camera Calibration Pipeline

End-to-end calibration of a Blickfeld QB2 LiDAR + RGB camera using
[direct_visual_lidar_calibration](https://github.com/koide3/direct_visual_lidar_calibration).

## Prerequisites

- Docker with NVIDIA GPU support (`nvidia-container-toolkit`)
- Input data:
  - `data/lidar/lidar_N.pcd` — binary PCD files (fields: x y z float32)
  - `data/lidar/lidar_N.npy` — Blickfeld frame objects with `photon_count` reflectivity
  - `data/camera/cam_N.png` — synchronized camera frames (1920×1080)
  - File numbers `N` must match across both sensors

## Step 0 — Build the Docker image (one time)

```bash
cd /research/nfs_ertin_1/home/yuyi/lidar_cal
docker build -f Dockerfile.superglue -t direct_visual_lidar_calibration:superglue .
```

The image extends `koide3/direct_visual_lidar_calibration:humble` with PyTorch,
SuperGluePretrainedNetwork, and the `blickfeld-qb2` SDK.

## Step 1 — Set camera intrinsics

Edit the constants at the top of `create_bags.py`:

```python
FX = 1248.8
FY = 1244.6
CX = 945.08
CY = 527.51
DIST = [k1, k2, 0.0, 0.0, 0.0]   # plumb_bob: k1, k2, p1, p2, k3
IMG_W, IMG_H = 1920, 1080
```

Obtain these from a checkerboard calibration (e.g. `cv2.calibrateCamera` or ROS camera_calibration).
Using placeholder values will produce a wrong translation in the final result.

## Step 2 — Create ROS2 bags

```bash
docker run --rm --gpus all 
  -v C:\Users\rose\OneDrive\Documents\0.Important_Files\2.School\3.Research\LidarDataAnalysis\data\yBags:/tmp/bags 
  -v C:\Users\rose\OneDrive\Documents\0.Important_Files\2.School\3.Research\LidarDataAnalysis\data\yData:/tmp/data:ro 
  -v C:\Users\rose\Documents\Lidar-Data-Analysis\direct_ros_cal\create_bags.py:/tmp/create_bags.py:ro 
  direct_visual_lidar_calibration:superglue 
  python3 /tmp/create_bags.py
```

Output: one ROS2 bag per frame pair under `bags/frame_NNN/`, each containing:
- `/image` (sensor_msgs/Image, rgb8)
- `/camera_info` (sensor_msgs/CameraInfo)
- `/points` (sensor_msgs/PointCloud2, fields: x y z intensity)

## Step 3 — Preprocess

```bash
docker run --rm --gpus all 
  -v C:\Users\rose\OneDrive\Documents\0.Important_Files\2.School\3.Research\LidarDataAnalysis\data\yBags:/tmp/input_bags 
  -v C:\Users\rose\OneDrive\Documents\0.Important_Files\2.School\3.Research\LidarDataAnalysis\data\ycustom_preprocessed:/tmp/preprocessed 
  direct_visual_lidar_calibration:superglue 
  ros2 run direct_visual_lidar_calibration preprocess /tmp/input_bags /tmp/preprocessed -a
```

Output per frame in ` /`:
- `frame_NNN.ply` — point cloud
- `frame_NNN.png` — camera image
- `frame_NNN_lidar_intensities.png` — LiDAR intensity image (will be replaced in Step 4)
- `frame_NNN_lidar_indices.png` — LiDAR point index map (will be replaced in Step 4)
- `calib.json` — intrinsics and metadata

## Step 4 — Fix LiDAR intensity images (Blickfeld-specific)

The default preprocess tool creates a full-sphere equirectangular image; the QB2's
~90°×49° FoV fills only ~5% of it. This step reprojects onto a FoV-matched canvas
and blends range + reflectivity to produce images with strong edges for matching.

```bash
docker run --rm --gpus all 
  -v C:\Users\rose\OneDrive\Documents\0.Important_Files\2.School\3.Research\LidarDataAnalysis\data\ycustom_preprocessed:/tmp/preprocessed 
  -v C:\Users\rose\Documents\Lidar-Data-Analysis\direct_ros_cal\fix_intensity_images.py:/tmp/fix_intensity_images.py:ro 
  direct_visual_lidar_calibration:superglue 
  python3 /tmp/fix_intensity_images.py
```

**Tuning** — edit `RANGE_WEIGHT` at the top of `fix_intensity_images.py`:
- `0.0` — pure reflectivity (photon_count only)
- `0.5` — equal blend of range and reflectivity (default)
- `1.0` — pure range (depth edges only)

Both `_lidar_intensities.png` and `_lidar_indices.png` are regenerated together so
they remain consistent. **Re-run this step every time preprocess is re-run.**

## Step 5 — SuperGlue feature matching

```bash
docker run --rm --gpus all -e MPLBACKEND=Agg 
  -v C:\Users\rose\OneDrive\Documents\0.Important_Files\2.School\3.Research\LidarDataAnalysis\data\ycustom_preprocessed:/tmp/preprocessed 
  direct_visual_lidar_calibration:superglue 
  ros2 run direct_visual_lidar_calibration find_matches_superglue.py /tmp/preprocessed
```

Output: `frame_NNN_matches.json` and `frame_NNN_superglue.png` (side-by-side match visualization).

Expected: ~100–300 matches per frame. Confidence scores are typically low (~0.05–0.1)
for cross-modal matching; this is normal. The calibration uses all matches, not just
high-confidence ones.

## Step 6 — Initial pose estimate

```bash
docker run --rm --gpus all 
  -v C:\Users\rose\OneDrive\Documents\0.Important_Files\2.School\3.Research\LidarDataAnalysis\data\ycustom_preprocessed:/tmp/preprocessed 
  direct_visual_lidar_calibration:superglue 
  ros2 run direct_visual_lidar_calibration initial_guess_auto /tmp/preprocessed
```

Writes `init_T_lidar_camera_auto` into `calib.json`. Requires Ceres to converge;
if it reports `NO_CONVERGENCE`, the match quality may be too low — try adjusting
`RANGE_WEIGHT` in Step 4 and re-running from Step 4.

## Step 7 — Fine calibration

```bash
docker run --rm --gpus all 
  -v C:\Users\rose\OneDrive\Documents\0.Important_Files\2.School\3.Research\LidarDataAnalysis\data\ycustom_preprocessed:/tmp/preprocessed 
  direct_visual_lidar_calibration:superglue 
  bash -c "export DISPLAY=192.168.1.7:0.0 && 
  ros2 run direct_visual_lidar_calibration calibrate /tmp/preprocessed --background --auto_quit"
```

Xvfb is required because the calibrate tool opens a GLFW window. Results are saved
to `calib.json` under `results.T_lidar_camera` **before** the viewer opens, so it is
safe to stop the container once the file is written.

`--auto_quit` exits automatically after convergence.

## Step 8 — Visualize results

```bash
docker run --rm --gpus all 
  -v C:\Users\rose\OneDrive\Documents\0.Important_Files\2.School\3.Research\LidarDataAnalysis\data\ycustom_preprocessed:/tmp/preprocessed 
  -v C:\Users\rose\Documents\Lidar-Data-Analysis\direct_ros_cal\visualize_cal.py:/tmp/visualize_cal.py:ro 
  direct_visual_lidar_calibration:superglue 
  python3 /tmp/visualize_cal.py /tmp/preprocessed
```

Output:
- `frame_NNN_projection.png` — LiDAR points projected onto the camera image, colored
  blue→red by range. Points should land on the correct surfaces/edges.
- `match_quality.png` — bar chart of per-frame SuperGlue match counts.

## Reading the result

`calib.json` after calibration:

```json
{
  "results": {
    "T_lidar_camera": [tx, ty, tz, qx, qy, qz, qw]
  }
}
```

`T_lidar_camera` transforms points **from camera frame to LiDAR frame**:

```
p_lidar = R(q) * p_camera + t
```

To project LiDAR points onto the camera image:

```python
T_camera_lidar = np.linalg.inv(T_lidar_camera)   # invert
p_camera = T_camera_lidar[:3,:3] @ p_lidar + T_camera_lidar[:3,3]
# then apply pinhole projection with fx, fy, cx, cy
```

## Known issues and workarounds

| Issue | Cause | Fix |
|---|---|---|
| LiDAR image nearly empty (std < 10) | Full-sphere projection, QB2 FoV is ~5% of sphere | Run Step 4 |
| LiDAR image horizontally mirrored | `arctan2(y,x)` increases right→left for +Y-forward sensor | Step 4 uses `az_max - az` mapping |
| `calibrate` NaN cost / jacobian failed | All LiDAR points behind camera (qx sign error) | Ensure `init_T_lidar_camera_auto` has qx < 0 |
| `rosbag2` directory exists error | Writer refuses pre-existing directory | `create_bags.py` does `shutil.rmtree` before writing |
| `numpy` version conflict in container | matplotlib upgrades numpy to 2.x, breaks cv2 | Dockerfile pins `numpy<2` |
| Translation result > 0.5 m | Wrong camera intrinsics | Set real fx/fy/cx/cy in Step 1 |


## Step 9 - Alternative
```bash
docker run --rm --gpus all 
  -v C:\Users\rose\OneDrive\Documents\0.Important_Files\2.School\3.Research\LidarDataAnalysis\data\yData:/tmp/data:ro 
  -v C:\Users\rose\OneDrive\Documents\0.Important_Files\2.School\3.Research\LidarDataAnalysis\data\yBags:/tmp/bags 
  -v C:\Users\rose\OneDrive\Documents\0.Important_Files\2.School\3.Research\LidarDataAnalysis\data\ycustom_preprocessed:/tmp/preprocessed 
  -v C:\Users\rose\Documents\Lidar-Data-Analysis\direct_ros_cal\:/tmp/scripts:ro 
  direct_visual_lidar_calibration:superglue 
  /tmp/scripts/run.sh
```

## Testing the accuracy of the transformation
Assuming you have some human made correlations between points in a point cloud and pixels in the corresponding image, it is possible to determine the average pixel projection error.  This is simply projecting points to pixels and determining the distance between the result and the expected result. 
```bash
docker run -it --rm --gpus all
  -v C:\Users\rose\OneDrive\Documents\0.Important_Files\2.School\3.Research\LidarDataAnalysis\data\ycustom_preprocessed:/tmp/preprocessed
  -v C:\Users\rose\Documents\Lidar-Data-Analysis\LiCamCal\match.json:/tmp/match.json:ro
  -v C:\Users\rose\Documents\Lidar-Data-Analysis\direct_ros_cal\test_accuracy.py:/tmp/scripts/test_accuracy.py:ro
  direct_visual_lidar_calibration:superglue 
  python3 -i /tmp/scripts/test_accuracy.py 

```