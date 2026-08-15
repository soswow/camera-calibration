# ROS / OpenCV camera calibration YAML

This project can export intrinsics in the **ROS `camera_info` YAML** format (also used by OpenCV tooling and `camera_calibration_parsers`). Produce it with:

```bash
python -m camera_calibration calibrate … \
  --output-folder output \
  --camera-name my_camera
```

That writes `output/my_camera-charuco-simple.json` and `output/my_camera-charuco-simple.yaml` (board and `--model` are part of the auto name). Or set `--output-name` for an explicit stem.

Or from an existing JSON result:

```python
from pathlib import Path
from camera_calibration import CalibrationResult

CalibrationResult.from_json(Path("output/intrinsics.json")).save_ros_yaml(
    Path("output/intrinsics.yaml"),
    camera_name="my_camera",
)
```

The file is plain YAML. Matrices are stored as OpenCV-style blocks with `rows`, `cols`, and a flat row-major `data` array.

## Example

```yaml
image_width: 2160
image_height: 3840
camera_name: pixel_1x
camera_matrix:
  rows: 3
  cols: 3
  data: [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [k1, k2, p1, p2, k3]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
# --- project-specific extension (ignored by stock ROS/OpenCV loaders) ---
fitzgibbon_lambda: -0.00058388
```

## Fields

### `image_width` / `image_height`

Sensor resolution **in pixels** for which the calibration was computed. Consumers should only apply these intrinsics to images of this size (or an explicitly scaled equivalent).

### `camera_name`

Human-readable label. Not used in the projection math. Set via `--camera-name` (YAML defaults to `camera` if omitted).

### `camera_matrix` — intrinsic matrix **K** (3×3)

Row-major layout of:

```text
[ fx,  0, cx ]
[  0, fy, cy ]
[  0,  0,  1 ]
```

| Symbol | Meaning |
| --- | --- |
| `fx`, `fy` | Focal length in pixels |
| `cx`, `cy` | Principal point (optical center) in pixels |

**Coordinate convention (OpenCV / ROS):** origin is the **top-left** of the image; `x` increases to the right; `y` increases downward. The geometric image center is approximately `(image_width/2, image_height/2)`; `cx`/`cy` are usually close to that but need not be exact.

### `distortion_model`

This exporter always writes `plumb_bob`, i.e. the Brown–Conrady / OpenCV “plumb bob” model.

Other ROS models exist (`rational_polynomial`, `equidistant`, …) but are not emitted here.

### `distortion_coefficients` — vector **D** (1×5)

For `plumb_bob`:

```text
[ k1, k2, p1, p2, k3 ]
```

| Coeff | Role |
| --- | --- |
| `k1`, `k2`, `k3` | Radial distortion |
| `p1`, `p2` | Tangential distortion |

OpenCV’s `calibrateCamera` / `undistort` use this same ordering. If a calibration run used a reduced model (e.g. `k1` only), unused coefficients are stored as `0.0`.

### `rectification_matrix` — **R** (3×3)

Rotation applied for **stereo rectification**. For a single (monocular) camera this project writes the **identity**:

```text
[ 1, 0, 0 ]
[ 0, 1, 0 ]
[ 0, 0, 1 ]
```

### `projection_matrix` — **P** (3×4)

Projects 3D points in the (rectified) camera frame into the image. For monocular, unrectified export this project writes **`[K | 0]`**:

```text
[ fx,  0, cx, 0 ]
[  0, fy, cy, 0 ]
[  0,  0,  1, 0 ]
```

The fourth column is the stereo baseline translation term (`Tx`, etc.). It is zero for a single camera.

In ROS image pipelines, undistorted/rectified topics are expected to be consistent with `P` (and `R`). For “load K and D and call `cv2.undistort`,” **`camera_matrix` + `distortion_coefficients` (+ image size)** are the essential fields.

### Project extension: Fitzgibbon λ

`fitzgibbon_lambda` is **not** part of stock ROS `CameraInfo`. Standard parsers typically **ignore unknown fields**, so adding it should not break ROS/OpenCV loaders that only read `K`/`D`/`R`/`P`.

| Field | Meaning |
| --- | --- |
| `fitzgibbon_lambda` | One-parameter division-model coefficient λ (Fitzgibbon 2001) |

Division model in the **normalized camera plane** (`z = 1`):

```text
x_d = x_u / (1 + λ * r_u²)
y_d = y_u / (1 + λ * r_u²)
r_u² = x_u² + y_u²
```

Then pixels are `u = fx * x_d + cx`, `v = fy * y_d + cy`.

Diagnostic RMS for the λ-only model (`fitzgibbon_rms_reprojection_error`) lives in the **JSON** / CLI report, not in this YAML.

## What most importers need

| Use case | Typically required |
| --- | --- |
| Undistort with OpenCV | `image_width/height`, `camera_matrix`, `distortion_coefficients` |
| ROS `image_proc` / rectify | Also `rectification_matrix`, `projection_matrix` |
| Identity / catalog only | `camera_name` |
| Division-model consumers (this project) | `fitzgibbon_lambda` |

## Relation to this project’s JSON

The JSON written by calibrate stores the same `K` and distortion coefficients (plus RMS, FOV helpers, used/rejected image lists, etc.). The YAML is the **interop** form; the JSON is the **full project result**.

Numerically:

- YAML `camera_matrix` ↔ JSON `camera_matrix`
- YAML `distortion_coefficients` ↔ JSON `distortion_coefficients` (first five entries, `plumb_bob`)
- YAML `image_width` / `image_height` ↔ JSON `image_size` `[width, height]`
- YAML `fitzgibbon_lambda` ↔ JSON `fitzgibbon_lambda`
- JSON-only: `fitzgibbon_rms_reprojection_error` (λ-only RMS; not exported to YAML)

## References

- ROS `sensor_msgs/CameraInfo` — calibration fields `K`, `D`, `R`, `P`
- ROS `camera_calibration_parsers` — YAML / INI read-write for the same layout
- OpenCV `calibrateCamera`, `undistort`, `getOptimalNewCameraMatrix` — consume `K` and `D` in this convention
- A. Fitzgibbon, *Simultaneous linear estimation of multiple view geometry and lens distortion*, ICCV 2001 — one-parameter division model
