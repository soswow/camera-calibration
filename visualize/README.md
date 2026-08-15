# Visualize

Plot camera matrix and distortion from a calibration JSON or ROS `camera_info` YAML: warped grid, undistort displacement, radial curve, and an optional before/after photo.

```bash
python -m camera_calibration visualize --help
python visualize/visualize.py --help   # same CLI (shim)
```

## Examples

```bash
python -m camera_calibration visualize output/intrinsics.json

python -m camera_calibration visualize output/intrinsics.yaml --exaggerate 10

python -m camera_calibration visualize output/intrinsics.json \
  --image path/to/photo.jpg \
  --output output/distortion-viz.png
```

## Options

| Option | Default / values | Description |
| --- | --- | --- |
| `calibration` | required | Intrinsics JSON or ROS `camera_info` YAML. |
| `--output` | `<calibration-stem>-viz.png` | PNG for the warped-grid / displacement / radial figure. |
| `--exaggerate` | `1` (`> 0`) | Scale the warped-grid offset only (heatmap and curves stay true). Use e.g. `10` when `k1` is tiny. |
| `--image` | unset | Optional photo at the calibrated resolution for a before/after PNG. |
| `--compare-output` | `<stem>-undistort-compare.png` | Where to write the before/after PNG (only with `--image`). |
| `--alpha` | `0` (`0`–`1`) | Undistort crop parameter for `--image`. |
