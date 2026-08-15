# Diagnose

Re-detect the board in a folder of photos and report coverage gaps, tilt/distance variety, and per-image reprojection error.

Board geometry is taken from the calibration JSON when present. Pass the board flags if you only have ROS YAML (no `pattern_size`).

```bash
python -m camera_calibration diagnose --help
python diagnose/diagnose.py --help   # same CLI (shim)
```

## Example

```bash
python -m camera_calibration diagnose path/to/images \
  --calibration output/intrinsics.json \
  --output output/diagnosis.json \
  --viz output/diagnosis.png
```

## Options

| Option | Default / values | Description |
| --- | --- | --- |
| `images` | required | Folder of calibration photos. |
| `--calibration` | required | Intrinsics JSON or ROS `camera_info` YAML. |
| `--output` | `output/diagnosis.json` | Diagnosis JSON path. |
| `--viz` | `output/diagnosis.png` | Visual summary PNG. |
| `--no-viz` | off | Skip the PNG. |
| `--top` | `10` | How many worst images to list in the text report. |
| `--board` | from JSON (`charuco`, `checkerboard`) | Override board type (needed for YAML with no pattern metadata). |
| `--squares-x` | from JSON | Override squares along X. |
| `--squares-y` | from JSON | Override squares along Y. |
| `--square-size` | from JSON | Override physical square edge. |
| `--marker-proportion` | from JSON, else `0.7` | **ChArUco only.** |
| `--dictionary` | from JSON, else `DICT_4X4_50` | **ChArUco only.** |
| `--min-charuco-corners` | `6` | **ChArUco only.** |
| `--detect-scale` | `0.35` | **Checkerboard only.** |
