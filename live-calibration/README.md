# Live ChArUco calibration

Interactive capture for collecting ChArUco observations in real time. Live preview with an optional coverage heatmap, frame acceptance thresholds, and periodic recalibration (reprojection error + coverage gaps).

Requires a PS3 Eye and the `ps3eye` module (see [ps3eye-tools/README.md](../ps3eye-tools/README.md)).

Demo: https://www.instagram.com/p/DUfEMg_k39c/

<img src="../docs/images/live_charuco_calibration/demo-screenshot.jpg" alt="Live ChArUco calibration preview" width="720">

No CLI flags; edit constants near the top of the script (board size, dictionary, acceptance thresholds, heatmap).

```bash
python live-calibration/live_charuco_calibration.py
```

Saved JSON (`output/live_calibration_latest.json` by default) can be passed to [`undistort`](../undistort/README.md) and [`visualize`](../visualize/README.md).
