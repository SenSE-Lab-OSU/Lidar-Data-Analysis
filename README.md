This is the code repository for the Lidar-Camera Extrinsic Calibration Project.  There are 3 methods of calibration in this repository: MATLAB App, Koide Toolbox, Manual Correspondance.   

## 1. MATLAB App
The MATLAB Lidar Camera Calibrator App is described in full in the *[documentation]((https://www.mathworks.com/help/lidar/ug/get-started-lidar-camera-calibrator.html))*.  Follow documentation instructions to generate a calibration.  Seems to result in good calibrations, however the automatic detection of PC Planes is unsucessful and manual selection is required.  It can be a challenge to manually select the planes properly and may take multiple attempts to get a result that MATLAB will accept.

`LiCamCal/ManualCalibration.m` - Follows the same steps as the App, but manual with less good GUI.  By using per-pair ROI for PC plane detection attempted to increase the probability of semi-automatic calibration.  Currently this method fails to produce acceptable calibrations, use the app and manually select planes instead.

`LiCamCal/` - Has other scripts that were used to visualize point clouds, cameras, transformations, etc.  May be useful, but not related to calibration
## 2. Koide Toolbox
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
  -v path/to/bags:/tmp/bags 
  -v path/to/data:/tmp/data:ro 
  -v path/to/create_bags.py:/tmp/create_bags.py:ro 
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
  -v path/to/bags:/tmp/input_bags 
  -v path/to/custom_preprocessed:/tmp/preprocessed 
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
  -v path/to/custom_preprocessed:/tmp/preprocessed 
  -v path/to/fix_intensity_images.py:/tmp/fix_intensity_images.py:ro 
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
  -v path/to/custom_preprocessed:/tmp/preprocessed 
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
  -v path/to/custom_preprocessed:/tmp/preprocessed 
  direct_visual_lidar_calibration:superglue 
  ros2 run direct_visual_lidar_calibration initial_guess_auto /tmp/preprocessed
```

Writes `init_T_lidar_camera_auto` into `calib.json`. Requires Ceres to converge;
if it reports `NO_CONVERGENCE`, the match quality may be too low — try adjusting
`RANGE_WEIGHT` in Step 4 and re-running from Step 4.

## Step 7 — Fine calibration

```bash
docker run --rm --gpus all 
  -v path/to/custom_preprocessed:/tmp/preprocessed 
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
  -v path/to/custom_preprocessed:/tmp/preprocessed 
  -v path/to/visualize_cal.py:/tmp/visualize_cal.py:ro 
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


## Step 9a - Alternative Running Previous steps in one go
```bash
docker run --rm --gpus all
  -v path/to/data:/tmp
  -v path/to/direct_ros_cal_new:/tmp/scripts:ro 
  direct_visual_lidar_calibration:superglue 
  /tmp/scripts/runOne.sh -i 172.28.7.66
```
Replace with your actual ip addr

## Step 9 - Alternative Running a grid search of 0-100% Range based lidar image
```bash
docker run --rm --gpus all 
  -v path/to/data:/tmp/data:ro 
  -v path/to/bags:/tmp/bags 
  -v path/to/custom_preprocessed:/tmp/preprocessed 
  -v path/to/direct_ros_cal:/tmp/scripts:ro 
  direct_visual_lidar_calibration:superglue 
  /tmp/scripts/runGrid.sh
```


## Testing the accuracy of the transformation
Assuming you have some human made correlations between points in a point cloud and pixels in the corresponding image, it is possible to determine the average pixel projection error.  This is simply projecting points to pixels and determining the distance between the result and the expected result. 
```bash
docker run -it --rm --gpus all
  -v path/to/custom_preprocessed:/tmp/preprocessed
  -v path/to/match.json:/tmp/match.json:ro
  -v path/to/test_accuracy.py:/tmp/scripts/test_accuracy.py:ro
  direct_visual_lidar_calibration:superglue 
  python3 -i /tmp/scripts/test_accuracy.py 

```
## 3. LiDAR–Camera Calibration Tools

Manual LiDAR-camera calibration, evaluation, and visualization. Four scripts:

| Script | Purpose |
|---|---|
| `calib_suite.py` | Interactive launcher that wraps the three tools below with session save/resume |
| `manual_calibrate.py` | Pick 2D↔3D correspondences and solve for the transform |
| `evaluate_calibration.py` | Reprojection error of one or more candidate transforms |
| `visualize_calibration.py` | Project a transform onto an image / point cloud |

**Requirements:** Python 3.10+, `numpy`, `opencv-python`, `scipy`, and optionally `matplotlib` (for evaluation plots).  -- `pip install -r requirements.txt` for all dependencies.

---

### Option A: `calib_suite.py` (recommended)

A menu-driven, fully interactive launcher — no arguments required beyond an optional session file. Keep it in the same folder as the three scripts above.

```bash
python calib_suite.py
```

Optional flags:

```bash
python calib_suite.py --session my_run.json      # load/create a specific session file
python calib_suite.py --scripts-dir /path/to/dir # if the 3 scripts live elsewhere
```

From the main menu you can:

- **New calibration from a data folder** — pick a folder, run the manual point-picking UI, and auto-register the resulting transform and correspondences.
- **Register an existing transform** — skip picking and import a transform JSON directly.
- **Evaluate transform(s)** — pick a correspondences file, then multi-select any number of registered transforms to compare via reprojection error.
- **Visualize a transform** — project a transform onto a single image/cloud pair or batch-process a directory.
- **Manage session** — rename/remove transforms and correspondence sets, change the output directory, view history.

File pickers use a native OS dialog when available, falling back to a text-based browser. All progress (data folder, registered transforms, correspondence sets, output paths) is saved to a session JSON file (default `./calib_session.json`), so you can quit anytime and resume exactly where you left off by re-running with the same `--session` path.

---

### Option B: Running the scripts individually

#### 1. `manual_calibrate.py`

Select corresponding points between camera images and LiDAR point clouds, then solve for the 6-DoF rigid transform.

**Data directory mode** (default):

```bash
python manual_calibrate.py [--data-dir ./data]
```

Expects:

```
<data-dir>/camera/cam_1.png   cam_2.png  ...
<data-dir>/lidar/lidar_1.npy  lidar_2.npy  ...
```

Files are matched by the trailing number in their names; only pairs with both a camera image and a LiDAR file are loaded.

**Single-pair mode:**

```bash
python manual_calibrate.py <camera_image> <lidar_npy>
```

**Controls:**

| Window | Control | Action |
|---|---|---|
| Camera | Left-click | Add 2D point |
| Camera | Right-click | Remove last 2D point |
| LiDAR | Left-click | Pick nearest 3D point |
| LiDAR | Right-click | Remove last 3D point |
| LiDAR | Scroll wheel | Zoom |
| LiDAR | Middle-drag | Pan |
| LiDAR | `T` / `F` / `S` | Top / Front / Side view |
| LiDAR | `G` | Toggle free-rotate mode |
| LiDAR | Arrow keys | Switch pair (←/→) or rotate 5° (all 4, when free-rotate is on) |
| Both | `[` / `]` | Previous / next pair |
| Both | `1`–`9` | Jump to pair N |
| Both | `ENTER` | Solve using all pairs combined |
| Both | `R` | Reset current pair's points |
| Both | `V` | Verify prior reprojection for current pair |
| Both | `Q` / `ESC` | Quit |

**Output:** `./calibration_result.json` (translation + quaternion + intrinsics) and `./correspondences.json` (all picked 2D/3D pairs, used by `evaluate_calibration.py`).

#### 2. `evaluate_calibration.py`

Re-projects the correspondences from `manual_calibrate.py` through one or more candidate transforms and reports per-point reprojection error.

```bash
python evaluate_calibration.py --correspondences correspondences.json \
    transform_A.json transform_B.json ... \
    [--labels "Method A" "Method B" ...] \
    [--out-csv errors.csv] \
    [--out-plot errors.png] \
    [--no-display]
```

**Arguments:**

- `correspondences.json` — output of `manual_calibrate.py`.
- `transform_*.json` (one or more) — each must contain either `"translation"` + `"quaternion"`, or a 4×4 row-major `"matrix"`. Intrinsics in the transform file are ignored; the correspondences file's intrinsics are always used.
- `--labels` — human-readable names for each transform (default: filenames).
- `--out-csv` — CSV of per-point errors (default `errors.csv`).
- `--out-plot` — save plots to this base path (suffixes `_boxplot`, `_summary`, `_cdf` are appended).
- `--no-display` — skip interactive plot windows (useful headless).

**Output:** console summary table, CSV of per-point errors, and (if `matplotlib` is installed) box plots, a summary bar chart, and CDF plots of reprojection error.

#### 3. `visualize_calibration.py`

Projects a transform two ways: LiDAR points onto the camera image (colored by depth), and the camera image's colors onto the LiDAR cloud (front view).

**Single-file mode:**

```bash
python visualize_calibration.py --paths <camera_image> <lidar_npy> -t <transform.json>
```

**Directory mode** — scans a folder for matched image/LiDAR files by filename keywords (`image`/`cam` for images, `lidar`/`cloud` for point clouds):

```bash
python visualize_calibration.py --dir <directory> -t <transform.json>
```

**Options:**

- `-t, --transform` (required) — transform JSON with `"translation"` + `"quaternion"`; may also embed camera intrinsics (`fx`, `fy`, `cx`, `cy`, `dist_coeffs`), otherwise defaults are used.
- `--no-display` — skip interactive windows, only save images.
- `--out-dir` — output directory (default: single mode = transform's folder; directory mode = `<dir>/visualizations/`).
- `--max-depth` — clip the depth color scale at this value (meters).
- `--radius` — point radius in pixels for the lidar-on-image render (default 2).

**Output:** `lidar_on_image.png` and `image_on_lidar_front.png` per pair, saved under the output directory.
