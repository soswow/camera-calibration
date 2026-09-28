# Single-camera fisheye calibration

The library accepts ordinary images of one camera. Perspective and fisheye
lenses use the same ChArUco/checkerboard input workflow. Rectangular and square
rasters are supported. Packed multi-camera images must be separated externally.

## Install and calibrate

```sh
python -m pip install -e .

camera-calibration calibrate path/to/images \
  --board charuco --squares-x 12 --squares-y 18 --square-size 40 \
  --marker-proportion 0.7 --dictionary DICT_4X4_250 \
  --model angular --validation-every 5 \
  --observations-out output/observations.json \
  --output-name my-camera
```

`--model simple` remains the default Brown model. `full` and `k1` remain Brown
presets. New models are `angular` and `angular-asymmetric`. The latter adds two
low-order angular-plane asymmetric terms; they are not Brown tangential
coefficients or photometric shading corrections.

Angular fitting needs at least eight distinct training views. The CLI reserves
every fifth distinct view by default and requires at least two validation views.
Vary board location, orientation and distance; include the rim. Partial ChArUco
boards are supported. A checkerboard still requires its complete inner grid.

`--radial-terms` selects 1–4 radial terms (default 2). `--seed-fov` is an initial
guess across the **shorter image dimension**, default 195°. `--max-angle` is the
mathematical half-angle limit, default 110°. For a narrower lens, set appropriate
values; these defaults do not measure or promise the lens's usable FOV. The seed
FOV must be smaller than twice the maximum angle. A larger polynomial is not
automatically better; the final mapping must remain invertible.

The default CLI `--max-rms 2` rejects fits exceeding 2 native pixels on training
or validation. It is adjustable, not a universal accuracy guarantee. Numerical
checks also reject failed optimization, parameter bounds and weakly constrained
angular fits. Low RMS does not establish coverage outside the photographed area.

## Stable validation and reusable observations

Saved observations contain original pixel positions, target coordinates, board
definition, image dimensions, source hashes when available, and detection metadata.
They contain no device layout. Reuse them without redetection:

```sh
camera-calibration calibrate path/to/images \
  --observations-in output/observations.json \
  --model angular-asymmetric --validation-list reserved.json \
  --output-name asymmetric-candidate
```

`reserved.json` is a JSON array of image names such as `["photo03.png",
"photo09.png"]`. It overrides periodic selection. Missing reserved detections
are an error. Exact image hashes and near-identical board observations are
grouped before splitting; a group containing a reserved image stays out of
training. Other duplicates are recorded as excluded. Save and reuse the
`validation.reserved_images` list from the profile when extending a dataset.
Automatic view selection runs only on the training partition.

The Python `calibrate_from_folder` API retains `validation_every=0` for existing
callers. Opt into the same partitioning with `validation_every=5` or `reserved`.
The CLI uses validation by default. Explicit `--validation-every 0` disables it;
such a fit has no independent validation score.

## Boards and masks

`generate-charuco` writes a `.board.json` sidecar containing exact board settings.
Use it instead of repeating flags:

```sh
camera-calibration calibrate path/to/images \
  --board-definition printed-board.board.json --model angular
```

The definition overrides board flags. Generation and detection share the same
board constructor, including `--legacy-pattern` and `--first-marker-id`.
Printing, PDF tiling and checkerboard support remain available.

ChArUco detection retains successful native detections. Failed native attempts
retry reduced scales with pixel-center-aware mapping and refinement on the
original raster. Temporary rotation retries map points back into the original
frame. Board consistency checks remain enabled.

For an unwanted second board, `--masks masks.json` accepts ordinary image names
mapped to exclusion polygons, with XY coordinates normalized to [0,1]:

```json
{
  "photo01.png": [[[0.7, 0.6], [1.0, 0.6], [1.0, 1.0], [0.7, 1.0]]],
  "photo02.png": []
}
```

Masks affect working detector images only; source files are not modified. Points
near the artificial boundary are excluded. The profile records masks for reuse
by `validate` and `diagnose`; an explicit masks file overrides them. Cached
observations already incorporate their masks and cannot be remasked at fit time.

## Validation, inspection and image output

```sh
camera-calibration validate path/to/reserved-images \
  --calibration output/my-camera.json --output output/validation.json

camera-calibration diagnose path/to/images \
  --calibration output/my-camera.json

camera-calibration visualize output/my-camera.json \
  --image path/to/photo.png --output output/model.png

camera-calibration undistort path/to/photo.png \
  --calibration output/my-camera.json --output output/perspective.png \
  --projection perspective --fov 100 --yaw 30 --pitch 0 --size 1200 800

camera-calibration undistort path/to/photo.png \
  --calibration output/my-camera.json --output output/spherical.png \
  --projection equirectangular --size 2048 1024
```

Reprojection RMS remains in original input pixels. For angular models, line
straightness is measured in degrees: bearings from each straight board row or
column should lie in a plane through the camera center. Validation overlays
orient a local perspective view toward the board, including at the rim. The
report warns about training-image reuse, including matching hashes under a
different filename. Previously rejected training images are not held-out data.

Diagnosis includes angular-sector corner counts and avoids demanding corners
outside a circular image. The model figure shows angular grid lines and radial
mapping/derivative. Its domain is a mathematical limit, not measured coverage.

Perspective output necessarily crops a >180° field. The default angular preview
uses 100° horizontal FOV. Yaw is positive toward camera-right; pitch is positive
toward camera-down. Equirectangular output uses a full sphere, with unsupported
directions black. Output validity checks model angle and source raster bounds;
they do not detect the physical image circle or certify measured calibration
support. `remap_view` also returns the validity array to Python callers. `alpha`
applies to the legacy Brown undistortion path, not angular output framing.

## Profiles and coordinates

New JSON profiles use schema version 2, explicitly identify the model and retain
pixel policy, board configuration, numerical checks, validation and provenance.
Image and camera axes are x right, y down, z forward. Pixel centers have integer
coordinates; no rotation or scaling is inferred from aspect ratio.

New calibrations use `pixel_policy: encoded`: EXIF display orientation is ignored,
and dimension mismatches are rejected. Old profiles lacking a policy retain the
historical inverse-EXIF/transpose behavior. `--pixel-policy legacy-exif` is available
for deliberate compatibility. Unknown profile versions and projection models
are rejected rather than interpreted as Brown.

Angular coefficients define:

```text
theta = atan2(hypot(X,Y), Z)
rho = theta * (1 + k1*theta² + k2*theta⁴ + ...)
(x,y) = rho * (X,Y) / hypot(X,Y)
```

The optional asymmetric correction acts on this angular plane:

```text
x' = x + 2*p1*x*y + p2*(3*x²+y²)
y' = y + p1*(x²+3*y²) + 2*p2*x*y
u = fx*x' + cx; v = fy*y' + cy
```

The radial inverse is bracketed over the declared domain. Asymmetric inversion
uses checked Newton iteration and a conservative sufficient invertibility bound.
No fisheye data passes through Brown projection/undistortion functions.

Angular profiles are JSON only. They cannot be silently exported as ROS
`plumb_bob`. Pinhole FOV and 35mm-equivalent fields are `null` for angular profiles;
K alone cannot supply those quantities. Existing Brown JSON/YAML workflows remain. Newly exported YAML includes a
`pixel_policy` extension so this library preserves its coordinate policy when
reloading; older YAML without it keeps legacy behavior.

The standalone PS3 Eye live application remains a Brown calibration application;
this merge does not add fisheye hardware capture, stereo calibration, HDR
stitching or shading estimation. Those applications can use the model API without
adding their device-specific inputs to this library.
