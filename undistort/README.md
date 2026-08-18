# Undistort

Remove lens distortion from one image or a folder, using saved intrinsics JSON or ROS `camera_info` YAML (including live ChArUco JSON).

```bash
python -m camera_calibration undistort --help
python undistort/undistort.py --help   # same CLI (shim)
```

## Examples

```bash
python -m camera_calibration undistort path/to/photo.jpg \
  --calibration output/intrinsics.json \
  --output output/undistorted/photo.jpg

python -m camera_calibration undistort path/to/folder \
  --calibration output/intrinsics.json \
  --output output/undistorted
```

## Options

| Option | Default / values | Description |
| --- | --- | --- |
| `source` | required | Image file or folder of images. |
| `--calibration` | `output/intrinsics.json` | Intrinsics JSON or ROS `camera_info` YAML. |
| `--output` | required | Output file (one image) or output folder (a folder of images). |
| `--alpha` | `0` (`0`–`1`) | New camera matrix scaling: `0` crops black borders, `1` keeps every source pixel. |

Source images are normalized with the same inverse EXIF orientation as
calibration and must then match the calibrated resolution exactly.
