# Calibrate

Estimate camera intrinsics from a folder of **ChArUco** or **checkerboard** photos. Always writes JSON and ROS/OpenCV YAML.

Count **squares** for both board types (`--squares-x` / `--squares-y`). Checkerboard detection still uses OpenCV inner corners internally (squares minus one); you do not pass that.

`--square-size` is the physical edge of one square (any unit, stay consistent). It only scales translations, not `fx`/`fy` in pixels. For a printed ChArUco board, measure a few squares with a ruler after printing.

```bash
python -m camera_calibration calibrate --help
python calibrate/calibrate.py --help   # same CLI (shim)
```

## Examples

```bash
python -m camera_calibration calibrate path/to/images \
  --board charuco \
  --squares-x 7 \
  --squares-y 10 \
  --square-size 79.14 \
  --marker-proportion 0.7 \
  --dictionary DICT_4X4_50 \
  --auto-select \
  --camera-name ps3eye \
  --preview-dir output/previews
```

Writes `output/ps3eye-charuco-simple.json` and `output/ps3eye-charuco-simple.yaml`.

```bash
python -m camera_calibration calibrate path/to/images \
  --board checkerboard \
  --squares-x 7 \
  --squares-y 10 \
  --square-size 25 \
  --auto-select \
  --output-folder output \
  --output-name my-lens
```

Writes `output/my-lens.json` and `output/my-lens.yaml`.

## Options

| Option | Default / values | Description |
| --- | --- | --- |
| `images` | required | Folder of calibration photos (non-recursive). |
| `--board` | required (`charuco`, `checkerboard`) | Which pattern is in the photos. |
| `--squares-x` | required (`>= 2`; checkerboard `>= 3`) | Squares along X. |
| `--squares-y` | required (`>= 2`; checkerboard `>= 3`) | Squares along Y. Provide both together. |
| `--square-size` | required (`> 0`) | Physical square edge. For printed ChArUco, measure after printing. |
| `--marker-proportion` | `0.7` (`(0, 1)`) | **ChArUco only.** Marker side as a fraction of square size. |
| `--dictionary` | `DICT_4X4_50` | **ChArUco only.** OpenCV ArUco dictionary name. |
| `--min-charuco-corners` | `6` (`>= 4`) | **ChArUco only.** Minimum interpolated corners to accept a view. |
| `--detect-scale` | `0.35` | **Checkerboard only.** Preferred downscale for corner detection on large photos. |
| `--model` | `simple` (`simple`, `full`, `k1`) | Distortion model. `simple` = k1,k2 only (stabler); `full` = k1..k3 + tangential (can overfit); `k1` = radial k1 only. |
| `--auto-select` | off | Fit, drop high-residual / hard close-tilt outliers while keeping pose diversity, then refit. |
| `--auto-select-max-keep` | unset (`>= 3`) | Cap on views kept after auto-select. |
| `--auto-select-error-factor` | `1.5` | Reject views with mean error &gt; factor × initial RMS. |
| `--auto-select-error-floor` | `2.0` px | Minimum error threshold for outlier rejection. |
| `--output-folder` | `output` | Directory for the JSON and YAML files. |
| `--output-name` | unset | Base filename (no extension) for `.json` and `.yaml`. If omitted and `--camera-name` is set, uses `<camera-name>-<board>-<model>`. Otherwise `intrinsics`. |
| `--camera-name` | unset (YAML field defaults to `camera`) | ROS YAML `camera_name`. When `--output-name` is omitted, also builds the output filename. |
| `--preview-dir` | unset | Write copies of images with detected corners drawn. |

Calibration reads encoded pixels and applies the inverse EXIF display
orientation before detecting corners. This lets phone portrait/landscape shots
share one camera pixel frame without guessing a rotation from width/height
alone. If an export has already baked the rotation into pixels and reset EXIF
Orientation to `1`, exact portrait/landscape transposes are still normalized to
the common calibration size. Images whose normalized dimensions still differ are
rejected from one calibration set.

Board detection also tries temporary 90-degree rotations when needed, then maps
detected corners back into the calibration frame. A board photographed sideways
is therefore valid; the temporary rotation is only for detection, not for the
intrinsics fit.

JSON includes `K`, Brown–Conrady `D`, RMS, FOV helpers, 35mm-equivalent focal length, used/failed image lists, and Fitzgibbon λ. ROS YAML format: [ros-camera-info-yaml.md](ros-camera-info-yaml.md).

## Tips

- Use 10–20 images with the board at different angles and distances, including frame edges and corners.
- Keep the board rigid and flat.
- If undistorted corners look worse, try `--model simple` (default) or `--model k1`.
- Ultrawide / fisheye lenses may need a fisheye model instead of pinhole + Brown–Conrady.
