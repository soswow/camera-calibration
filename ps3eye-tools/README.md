# PS3 Eye preview tools

Quick sanity checks with a PS3 Eye: confirm device access, dial in exposure, and judge focus before calibration. No CLI options; edit constants near the top of each file.

## Focus helper (`test_ps3eye_focus.py`)

<img src="../docs/images/ps3eye_tools/sharp_tool_demo.png" alt="Focus tool example" width="720">

Detects a checkerboard and shows a rolling sharpness score. It locks onto the checkerboard region after a brief locking period, then tracks sharpness in that region. Square counts are `CHECKERBOARD_SQUARES_X` / `CHECKERBOARD_SQUARES_Y` at the top of the file.

```bash
python ps3eye-tools/test_ps3eye_focus.py
```

## Preview (`test_ps3eye_preview.py`)

Minimal live view to verify the camera is streaming and to tune exposure/gain/FPS.

```bash
python ps3eye-tools/test_ps3eye_preview.py
```

## Dependency: `ps3eye` Python module

Build from https://github.com/soswow/PS3Eye-library :

```bash
python -m pip install pybind11 numpy
cmake -S . -B build -DBUILD_PYTHON=ON -DCMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir)"
cmake --build build
```

Then add the build directory to `PYTHONPATH`:

```bash
export PYTHONPATH="/path/to/PS3Eye-Driver-MacOS-Silicon/build:${PYTHONPATH}"
```

You can also run these scripts from that repo’s `build/` directory so the module is discoverable.
