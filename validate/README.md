# Validate

Measure genuine held-out reprojection error and undistorted grid straightness
using a saved calibration. The command accepts one board image or a folder.
Saved intrinsics remain fixed; only the pose of each validation board is fitted.

```bash
python -m camera_calibration validate path/to/held-out.jpg \
  --calibration output/intrinsics.json \
  --output output/validation.json \
  --viz output/validation-viz.png
```

Board geometry and ChArUco dictionary metadata are read from project JSON.
They can be overridden with the usual `--board`, `--squares-x`, `--squares-y`,
`--square-size`, `--marker-proportion`, and `--dictionary` options. ROS YAML has
no board geometry, so those overrides are required when validating from YAML.

The report contains pooled and per-image:

- true reprojection RMS with fixed intrinsics;
- mean, median, p95, and maximum corner error;
- undistorted row/column straightness RMS, median, p95, and maximum;
- warnings when a supposed validation image appears in `used_images` or
  `auto_select_rejected` and therefore is not genuinely held out.

Unless `--no-viz` is passed, validation also writes an undistorted overlay with
semi-transparent fitted straight lines: amber for board rows, blue for columns,
and white dots for the detected intersections. For one input image, `--viz` is
the output PNG. For a folder, it is an output directory. Use `--line-opacity`
to adjust the lines and `--alpha` to choose cropped (`0`) or full-FOV (`1`)
undistortion.
