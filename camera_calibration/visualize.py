"""Visualize calibrated intrinsics and lens distortion."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .images import orient_image_to_size
from .result import CalibrationResult
from .undistort import undistort_image


def _camera_arrays(
    calibration: CalibrationResult,
) -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.asarray(calibration.camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(calibration.distortion_coefficients, dtype=np.float64).reshape(-1)
    return camera_matrix, dist_coeffs


def _brown_coeffs(dist_coeffs: np.ndarray) -> tuple[float, float, float, float, float]:
    padded = np.zeros(5, dtype=np.float64)
    padded[: min(5, dist_coeffs.size)] = dist_coeffs[:5]
    k1, k2, p1, p2, k3 = padded.tolist()
    return k1, k2, p1, p2, k3


def distort_points(
    undistorted_pixels: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """
    Apply Brown–Conrady (OpenCV) distortion to undistorted pixel coordinates.

    `undistorted_pixels` is (N, 2) in the ideal pinhole image. Result is where
    those rays hit the sensor.
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    xy = undistorted_pixels.reshape(-1, 2).astype(np.float64)
    # Lift pixels to the camera plane (z=1), then projectPoints applies D and K.
    object_points = np.column_stack(
        (
            (xy[:, 0] - cx) / fx,
            (xy[:, 1] - cy) / fy,
            np.ones(len(xy)),
        )
    ).reshape(-1, 1, 3)
    projected, _ = cv2.projectPoints(
        object_points,
        np.zeros(3),
        np.zeros(3),
        camera_matrix,
        dist_coeffs,
    )
    return projected.reshape(-1, 2)


def _grid_polylines(width: int, height: int, steps: int = 13, samples: int = 80) -> list[np.ndarray]:
    """Regular pixel-space grid as dense polylines (ideal / undistorted).

    Outer lines are inset so stroke width cannot paint over the axis spines.
    """
    inset = max(width, height) * 0.006
    xs = np.linspace(inset, width - inset, steps)
    ys = np.linspace(inset, height - inset, steps)
    t_x = np.linspace(inset, width - inset, samples)
    t_y = np.linspace(inset, height - inset, samples)
    lines: list[np.ndarray] = []
    for x in xs:
        lines.append(np.column_stack((np.full(samples, x), t_y)))
    for y in ys:
        lines.append(np.column_stack((t_x, np.full(samples, y))))
    return lines


def undistort_displacement_maps(
    calibration: CalibrationResult,
    max_edge: int = 640,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsampled undistort geometry in original-resolution pixels.

    Returns dest_x, dest_y, dx, dy, magnitude. OpenCV remap samples
    undistorted (u, v) from the distorted image at (u+dx, v+dy).
    """
    width, height = calibration.image_size
    scale = min(1.0, max_edge / float(max(width, height)))
    map_width = max(2, int(round(width * scale)))
    map_height = max(2, int(round(height * scale)))
    camera_matrix, dist_coeffs = _camera_arrays(calibration)
    # Same FOV at a smaller raster: scale fx, fy, cx, cy with the image size.
    scaled_k = camera_matrix.copy()
    scaled_k[0, 0] *= scale
    scaled_k[1, 1] *= scale
    scaled_k[0, 2] *= scale
    scaled_k[1, 2] *= scale
    map_x, map_y = cv2.initUndistortRectifyMap(
        scaled_k,
        dist_coeffs,
        None,
        scaled_k,
        (map_width, map_height),
        cv2.CV_32FC1,
    )
    grid_x, grid_y = np.meshgrid(
        np.arange(map_width, dtype=np.float64),
        np.arange(map_height, dtype=np.float64),
    )
    # Convert downsampled remap offset back to original-resolution pixels.
    dx = (map_x - grid_x) / scale
    dy = (map_y - grid_y) / scale
    dest_x = grid_x / scale
    dest_y = grid_y / scale
    magnitude = np.hypot(dx, dy)
    return dest_x, dest_y, dx, dy, magnitude


def _radial_profiles(
    calibration: CalibrationResult,
    samples: int = 400,
) -> dict[str, np.ndarray]:
    """Radial mapping vs normalized radius, plus pixel displacement."""
    camera_matrix, dist_coeffs = _camera_arrays(calibration)
    k1, k2, _p1, _p2, k3 = _brown_coeffs(dist_coeffs)
    width, height = calibration.image_size
    fx, fy = calibration.fx, calibration.fy
    cx, cy = calibration.cx, calibration.cy
    corners = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [0.0, height - 1.0],
            [width - 1.0, height - 1.0],
        ]
    )
    x_n = (corners[:, 0] - cx) / fx
    y_n = (corners[:, 1] - cy) / fy
    r_corner = float(np.max(np.hypot(x_n, y_n)))
    r_u = np.linspace(0.0, r_corner, samples)
    r2 = r_u * r_u
    # Tangential terms are not purely radial; this curve is the radial part only.
    radial_scale = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    r_d_brown = r_u * radial_scale
    mean_f = 0.5 * (fx + fy)
    delta_px_brown = mean_f * (r_d_brown - r_u)

    result = {
        "r_u": r_u,
        "r_corner": np.array([r_corner]),
        "delta_px_brown": delta_px_brown,
        "r_half_w": np.array([abs((width / 2.0) / fx)]),
        "r_half_h": np.array([abs((height / 2.0) / fy)]),
    }
    if calibration.fitzgibbon_lambda is not None:
        denom = 1.0 + calibration.fitzgibbon_lambda * r2
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        r_d_fitz = r_u / denom
        result["delta_px_fitz"] = mean_f * (r_d_fitz - r_u)
    return result


def _barrel_label(k1: float) -> str:
    if k1 < -1e-8:
        return "barrel (k1 < 0)"
    if k1 > 1e-8:
        return "pincushion (k1 > 0)"
    return "near-zero radial k1"


def _lock_image_axes(axis, width: int, height: int):
    """Keep the frame at the sensor rectangle; warped lines must not expand the view."""
    from matplotlib.patches import Rectangle

    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_aspect("equal", adjustable="box")
    axis.autoscale(enable=False)
    axis.set_axisbelow(False)
    for spine in axis.spines.values():
        spine.set_zorder(20)
    frame = Rectangle((0, 0), width, height, fill=False, edgecolor="0.4", linewidth=1.0, zorder=21)
    axis.add_patch(frame)
    return frame


def _liang_barsky(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> tuple[float, float, float, float] | None:
    """Clip one segment to an axis-aligned rectangle. None if fully outside."""
    dx = x1 - x0
    dy = y1 - y0
    p_values = (-dx, dx, -dy, dy)
    q_values = (x0 - x_min, x_max - x0, y0 - y_min, y_max - y0)
    u1 = 0.0
    u2 = 1.0
    for p_value, q_value in zip(p_values, q_values):
        if abs(p_value) < 1e-18:
            if q_value < 0:
                return None
            continue
        t_value = q_value / p_value
        if p_value < 0:
            if t_value > u2:
                return None
            if t_value > u1:
                u1 = t_value
        else:
            if t_value < u1:
                return None
            if t_value < u2:
                u2 = t_value
    return (x0 + u1 * dx, y0 + u1 * dy, x0 + u2 * dx, y0 + u2 * dy)


def _clip_polyline_to_rect(
    points: np.ndarray,
    width: float,
    height: float,
) -> list[np.ndarray]:
    """Split a polyline into the pieces that lie inside the sensor rectangle."""
    pieces: list[np.ndarray] = []
    current: list[tuple[float, float]] = []
    for index in range(len(points) - 1):
        x0, y0 = float(points[index, 0]), float(points[index, 1])
        x1, y1 = float(points[index + 1, 0]), float(points[index + 1, 1])
        clipped = _liang_barsky(x0, y0, x1, y1, 0.0, 0.0, width, height)
        if clipped is None:
            if len(current) >= 2:
                pieces.append(np.asarray(current, dtype=np.float64))
            current = []
            continue
        cx0, cy0, cx1, cy1 = clipped
        if not current:
            current.append((cx0, cy0))
        elif abs(current[-1][0] - cx0) > 1e-6 or abs(current[-1][1] - cy0) > 1e-6:
            if len(current) >= 2:
                pieces.append(np.asarray(current, dtype=np.float64))
            current = [(cx0, cy0)]
        current.append((cx1, cy1))
    if len(current) >= 2:
        pieces.append(np.asarray(current, dtype=np.float64))
    return pieces


def _plot_clipped(axis, xs, ys, width: int, height: int, **plot_kwargs):
    """Draw a polyline clipped to the sensor rectangle."""
    points = np.column_stack((np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)))
    pieces = _clip_polyline_to_rect(points, float(width), float(height))
    line = None
    for piece in pieces:
        (line,) = axis.plot(piece[:, 0], piece[:, 1], **plot_kwargs)
    return line


def _format_calibration_summary(
    calibration: CalibrationResult,
    k1: float,
    k2: float,
    p1: float,
    p2: float,
    k3: float,
) -> str:
    width, height = calibration.image_size
    lines = [
        "Intrinsics",
        f"  fx          {calibration.fx:.4f}",
        f"  fy          {calibration.fy:.4f}",
        f"  cx          {calibration.cx:.4f}",
        f"  cy          {calibration.cy:.4f}",
        f"  Δcenter     ({calibration.cx - width / 2.0:+.2f}, "
        f"{calibration.cy - height / 2.0:+.2f}) px",
        f"  size        {width} × {height}",
        "",
        "Field of view",
        f"  HFOV        {calibration.hfov_deg:.2f}°",
        f"  VFOV        {calibration.vfov_deg:.2f}°",
        f"  DFOV        {calibration.dfov_deg:.2f}°",
        f"  35mm equiv. {calibration.focal_length_35mm_equiv:.1f} mm",
        "",
        "Brown–Conrady",
        f"  model       {calibration.distortion_model}",
        f"  k1          {k1:.6g}",
        f"  k2          {k2:.6g}",
        f"  k3          {k3:.6g}",
        f"  p1          {p1:.6g}",
        f"  p2          {p2:.6g}",
    ]
    if calibration.rms_reprojection_error:
        lines.append(f"  RMS         {calibration.rms_reprojection_error:.4f} px")
    if calibration.fitzgibbon_lambda is not None:
        lines.extend(
            [
                "",
                "Fitzgibbon division",
                f"  λ           {calibration.fitzgibbon_lambda:.6g}",
            ]
        )
        if calibration.fitzgibbon_rms_reprojection_error:
            lines.append(
                f"  RMS         {calibration.fitzgibbon_rms_reprojection_error:.4f} px"
            )
    return "\n".join(lines)


def render_distortion_figure(
    calibration: CalibrationResult,
    path: Path,
    exaggerate: float = 1.0,
) -> Path:
    """
    Write a 2×2 plot grid plus a right-hand parameter panel.

    1) Ideal grid vs Brown–Conrady warped grid (what a straight scene does on the sensor)
    2) Undistort displacement heatmap (how far remap moves pixels)
    3) Radial Δr in pixels vs normalized radius
    4) Downsampled displacement quiver
    5) Printed K / D / FOV / λ
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if exaggerate <= 0:
        raise ValueError("exaggerate must be positive")

    width, height = calibration.image_size
    camera_matrix, dist_coeffs = _camera_arrays(calibration)
    k1, k2, p1, p2, k3 = _brown_coeffs(dist_coeffs)
    dest_x, dest_y, disp_dx, disp_dy, magnitude = undistort_displacement_maps(calibration)
    max_disp = float(np.nanmax(magnitude))
    profiles = _radial_profiles(calibration)

    figure = plt.figure(figsize=(17.5, 11), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.62])
    axis_grid = figure.add_subplot(grid[0, 0])
    axis_heat = figure.add_subplot(grid[0, 1])
    axis_radial = figure.add_subplot(grid[1, 0])
    axis_quiver = figure.add_subplot(grid[1, 1])
    axis_meta = figure.add_subplot(grid[:, 2])
    figure.suptitle(
        f"Lens / intrinsics  ·  {width}×{height}  ·  "
        f"max undistort shift {max_disp:.2f} px  ·  {_barrel_label(k1)}",
        fontsize=13,
    )

    # --- Panel 1: warped grid ---
    _lock_image_axes(axis_grid, width, height)
    for polyline in _grid_polylines(width, height):
        _plot_clipped(
            axis_grid,
            polyline[:, 0],
            polyline[:, 1],
            width,
            height,
            color="0.82",
            linewidth=0.7,
            solid_capstyle="butt",
            clip_on=True,
            zorder=2,
        )
        warped = distort_points(polyline, camera_matrix, dist_coeffs)
        if exaggerate != 1.0:
            warped = polyline + exaggerate * (warped - polyline)
        _plot_clipped(
            axis_grid,
            warped[:, 0],
            warped[:, 1],
            width,
            height,
            color="#c0392b",
            linewidth=1.1,
            solid_capstyle="butt",
            clip_on=True,
            zorder=3,
        )
    axis_grid.set_xlim(0, width)
    axis_grid.set_ylim(height, 0)
    axis_grid.scatter(
        [width / 2.0, calibration.cx],
        [height / 2.0, calibration.cy],
        c=["0.45", "#1f77b4"],
        s=[36, 42],
        zorder=5,
        edgecolors="white",
        linewidths=0.4,
        clip_on=True,
    )
    axis_grid.set_title(
        "Straight grid → sensor (red)"
        + (f", ×{exaggerate:g} warp" if exaggerate != 1.0 else "")
    )
    axis_grid.set_xlabel("x (px)")
    axis_grid.set_ylabel("y (px)")
    axis_grid.plot([], [], color="0.82", label="ideal pinhole")
    axis_grid.plot([], [], color="#c0392b", label="Brown–Conrady")
    axis_grid.scatter([], [], c="#1f77b4", label="principal point")
    axis_grid.scatter([], [], c="0.45", label="geometric center")
    axis_grid.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # --- Panel 2: displacement heatmap ---
    heatmap = axis_heat.imshow(
        magnitude,
        cmap="magma",
        origin="upper",
        extent=(0, width, height, 0),
        aspect="equal",
        vmin=0.0,
    )
    axis_heat.scatter(
        [calibration.cx],
        [calibration.cy],
        c="#7fd0ff",
        s=36,
        zorder=5,
        edgecolors="white",
        linewidths=0.4,
    )
    axis_heat.set_title("Undistort displacement |source − dest| (px)")
    axis_heat.set_xlabel("x (px)")
    axis_heat.set_ylabel("y (px)")
    figure.colorbar(heatmap, ax=axis_heat, fraction=0.046, pad=0.04, label="pixels")

    # --- Panel 3: radial curve ---
    r_u = profiles["r_u"]
    axis_radial.axhline(0.0, color="0.75", linewidth=0.8)
    axis_radial.axvline(
        float(profiles["r_half_w"][0]),
        color="0.8",
        linestyle="--",
        linewidth=0.8,
        label="half width",
    )
    axis_radial.axvline(
        float(profiles["r_half_h"][0]),
        color="0.8",
        linestyle=":",
        linewidth=0.8,
        label="half height",
    )
    axis_radial.axvline(
        float(profiles["r_corner"][0]),
        color="0.55",
        linestyle="-",
        linewidth=0.8,
        label="corner",
    )
    axis_radial.plot(
        r_u, profiles["delta_px_brown"], color="#c0392b", linewidth=2.0, label="Brown–Conrady radial"
    )
    if "delta_px_fitz" in profiles:
        axis_radial.plot(
            r_u,
            profiles["delta_px_fitz"],
            color="#1f77b4",
            linewidth=1.6,
            linestyle="--",
            label="Fitzgibbon λ",
        )
    axis_radial.set_title("Radial shift vs normalized radius (tangential ignored)")
    axis_radial.set_xlabel("r_u  (normalized, at z=1)")
    axis_radial.set_ylabel("Δr · mean(fx,fy)  (px)")
    axis_radial.legend(loc="best", fontsize=8)
    axis_radial.grid(True, alpha=0.25)

    # --- Panel 4: quiver ---
    _lock_image_axes(axis_quiver, width, height)
    step = max(1, min(dest_x.shape) // 16)
    axis_quiver.quiver(
        dest_x[::step, ::step],
        dest_y[::step, ::step],
        disp_dx[::step, ::step],
        disp_dy[::step, ::step],
        magnitude[::step, ::step],
        cmap="magma",
        angles="xy",
        # Length is autoscaled so small warps stay visible; color is true px.
        width=0.003,
        pivot="tail",
        clip_on=True,
    )
    axis_quiver.set_title("Undistort flow (arrow length scaled for visibility)")
    axis_quiver.set_xlabel("x (px)")
    axis_quiver.set_ylabel("y (px)")
    axis_quiver.set_xlim(0, width)
    axis_quiver.set_ylim(height, 0)

    # --- Panel 5: parameters ---
    axis_meta.set_axis_off()
    axis_meta.set_xlim(0, 1)
    axis_meta.set_ylim(0, 1)
    axis_meta.set_title("Calibration")
    axis_meta.text(
        0.04,
        0.98,
        _format_calibration_summary(calibration, k1, k2, p1, p2, k3),
        transform=axis_meta.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        linespacing=1.35,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def render_undistort_comparison(
    calibration: CalibrationResult,
    image_path: Path,
    output_path: Path,
    alpha: float = 0.0,
) -> Path:
    """Side-by-side captured frame vs undistorted, with the warped/straight grids overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    oriented = orient_image_to_size(image, calibration.image_size)
    if oriented is None:
        raise ValueError(
            f"Image size does not match calibration {calibration.image_size[0]}x"
            f"{calibration.image_size[1]} (and is not a portrait/landscape transpose)"
        )
    image, _was_rotated = oriented
    undistorted = undistort_image(image, calibration, alpha=alpha)
    camera_matrix, dist_coeffs = _camera_arrays(calibration)
    width, height = calibration.image_size

    rgb_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_undistorted = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)

    figure, axes = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)
    figure.suptitle(f"Undistort preview  ·  {image_path.name}  ·  alpha={alpha:g}")

    axes[0].imshow(rgb_original)
    axes[0].set_title("Captured + straight-grid-as-seen-by-lens")
    for polyline in _grid_polylines(width, height, steps=11, samples=60):
        warped = distort_points(polyline, camera_matrix, dist_coeffs)
        for piece in _clip_polyline_to_rect(warped, float(width), float(height)):
            axes[0].plot(piece[:, 0], piece[:, 1], color="#ffee58", linewidth=0.7, alpha=0.85)

    axes[1].imshow(rgb_undistorted)
    axes[1].set_title("After undistort (look for straightened lines)")

    for axis in axes:
        axis.set_axis_off()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=120)
    plt.close(figure)
    return output_path
