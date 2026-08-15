"""Diagnose calibration coverage and per-image reprojection quality."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .detection import DetectionSet
from .result import CalibrationResult


@dataclass
class ImageDiagnosis:
    name: str
    mean_error_px: float
    max_error_px: float
    # Board center in normalized image coords [0,1]x[0,1]
    center_x: float
    center_y: float
    # Fraction of image width/height spanned by board bbox
    span_x: float
    span_y: float
    # Camera-to-board distance in square_size units (from tvec)
    distance: float
    # Board tilt: angle between board normal and optical axis (degrees)
    tilt_deg: float
    # Roll around optical axis-ish (degrees), from Rodrigues
    roll_deg: float
    quadrant: str  # TL/TR/BL/BR/center
    corner_count: int


@dataclass
class CoverageGap:
    code: str
    severity: str  # high | medium | low
    message: str


@dataclass
class DiagnosisReport:
    image_size: tuple[int, int]
    pattern_size: tuple[int, int]
    square_size: float
    distortion_model: str
    overall_rms: float
    images: list[ImageDiagnosis]
    failed_images: list[str]
    # 4x4 grid of corner hit counts (row-major, top-left first)
    spatial_grid: list[list[int]]
    grid_rows: int
    grid_cols: int
    gaps: list[CoverageGap] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def _quadrant(center_x: float, center_y: float) -> str:
    # Center band if within middle 40% of both axes.
    if 0.3 <= center_x <= 0.7 and 0.3 <= center_y <= 0.7:
        return "center"
    vertical = "T" if center_y < 0.5 else "B"
    horizontal = "L" if center_x < 0.5 else "R"
    return f"{vertical}{horizontal}"


def _rotation_tilt_roll(rvec: np.ndarray) -> tuple[float, float]:
    """Return (tilt_deg, roll_deg) from a Rodrigues rotation vector."""
    rotation, _ = cv2.Rodrigues(rvec)
    # Board normal in camera coords (board Z axis).
    normal = rotation[:, 2]
    optical = np.array([0.0, 0.0, 1.0])
    cos_tilt = float(np.clip(abs(np.dot(normal, optical)), -1.0, 1.0))
    tilt_deg = math.degrees(math.acos(cos_tilt))
    # Roll: orientation of board X projected onto image plane.
    board_x = rotation[:, 0]
    roll_deg = math.degrees(math.atan2(board_x[1], board_x[0]))
    return tilt_deg, roll_deg


def _accumulate_spatial_grid(
    corners: np.ndarray,
    image_size: tuple[int, int],
    grid: np.ndarray,
) -> None:
    width, height = image_size
    rows, cols = grid.shape
    points = corners.reshape(-1, 2)
    for x, y in points:
        col = min(cols - 1, max(0, int(x / width * cols)))
        row = min(rows - 1, max(0, int(y / height * rows)))
        grid[row, col] += 1


def _assess_gaps(
    images: list[ImageDiagnosis],
    spatial_grid: np.ndarray,
    overall_rms: float,
) -> tuple[list[CoverageGap], dict]:
    gaps: list[CoverageGap] = []
    if not images:
        return gaps, {}

    n = len(images)
    centers_x = [image.center_x for image in images]
    centers_y = [image.center_y for image in images]
    distances = [image.distance for image in images]
    tilts = [image.tilt_deg for image in images]
    spans = [max(image.span_x, image.span_y) for image in images]
    mean_errors = [image.mean_error_px for image in images]

    quadrant_counts = {
        key: sum(1 for image in images if image.quadrant == key)
        for key in ("TL", "TR", "BL", "BR", "center")
    }

    # Edge/corner frame coverage via board centers.
    near_left = sum(1 for value in centers_x if value < 0.25)
    near_right = sum(1 for value in centers_x if value > 0.75)
    near_top = sum(1 for value in centers_y if value < 0.25)
    near_bottom = sum(1 for value in centers_y if value > 0.75)

    empty_cells = int(np.sum(spatial_grid == 0))
    total_cells = int(spatial_grid.size)
    corner_cells = [
        spatial_grid[0, 0],
        spatial_grid[0, -1],
        spatial_grid[-1, 0],
        spatial_grid[-1, -1],
    ]
    empty_corners = sum(1 for count in corner_cells if count == 0)

    tilt_high = sum(1 for value in tilts if value >= 30)
    tilt_med = sum(1 for value in tilts if 15 <= value < 30)
    tilt_low = sum(1 for value in tilts if value < 15)

    dist_min, dist_max = min(distances), max(distances)
    dist_ratio = dist_max / max(dist_min, 1e-6)

    high_error = [image for image in images if image.mean_error_px > max(2.0, overall_rms * 1.5)]
    high_error_fraction = len(high_error) / n

    if overall_rms > 1.5:
        gaps.append(
            CoverageGap(
                code="high_rms",
                severity="high",
                message=(
                    f"Overall RMS is {overall_rms:.2f} px (target usually ≤ 1.0). "
                    "Coverage gaps or a few bad views are likely dominating."
                ),
            )
        )

    if empty_corners >= 2:
        gaps.append(
            CoverageGap(
                code="missing_frame_corners",
                severity="high",
                message=(
                    f"{empty_corners}/4 image-corner regions have almost no detected "
                    "board corners. Distortion is poorly constrained there — "
                    "put the board near each frame corner in some shots."
                ),
            )
        )
    elif empty_cells / total_cells > 0.35:
        gaps.append(
            CoverageGap(
                code="sparse_spatial_coverage",
                severity="medium",
                message=(
                    f"{empty_cells}/{total_cells} spatial bins have no corners. "
                    "Spread the board across more of the frame."
                ),
            )
        )

    for label, count in (
        ("left edge", near_left),
        ("right edge", near_right),
        ("top edge", near_top),
        ("bottom edge", near_bottom),
    ):
        if count == 0:
            gaps.append(
                CoverageGap(
                    code=f"missing_{label.replace(' ', '_')}",
                    severity="medium",
                    message=f"No board centers near the {label}. Add a few shots there.",
                )
            )

    missing_quadrants = [name for name, count in quadrant_counts.items() if name != "center" and count == 0]
    if missing_quadrants:
        gaps.append(
            CoverageGap(
                code="missing_quadrants",
                severity="medium",
                message=f"No board centers in quadrant(s): {', '.join(missing_quadrants)}.",
            )
        )

    if tilt_high < max(3, n // 8):
        gaps.append(
            CoverageGap(
                code="low_tilt_variety",
                severity="high" if tilt_high == 0 else "medium",
                message=(
                    f"Only {tilt_high} views with tilt ≥ 30° (have {tilt_low} nearly-frontal). "
                    "Strong perspective (board tilted toward camera) is needed to pin down focal length."
                ),
            )
        )

    if dist_ratio < 1.4:
        gaps.append(
            CoverageGap(
                code="narrow_distance_range",
                severity="medium",
                message=(
                    f"Board distance only spans {dist_min:.1f}–{dist_max:.1f} "
                    f"(ratio {dist_ratio:.2f}). Mix closer and farther poses."
                ),
            )
        )

    mostly_huge = sum(1 for span in spans if span > 0.7) / n
    if mostly_huge > 0.7:
        gaps.append(
            CoverageGap(
                code="mostly_closeups",
                severity="medium",
                message=(
                    f"{mostly_huge:.0%} of views fill >70% of the frame. "
                    "Close-ups under-sample edges; add mid/far placements."
                ),
            )
        )

    center_heavy = quadrant_counts["center"] / n
    if center_heavy > 0.5:
        gaps.append(
            CoverageGap(
                code="center_heavy",
                severity="medium",
                message=(
                    f"{center_heavy:.0%} of board centers sit in the middle of the frame. "
                    "Even if corners get some hits, bias toward mid-frame weakens "
                    "edge/corner distortion constraints — add more off-center placements."
                ),
            )
        )

    # Close + steeply tilted views often dominate RMS even on a rigid board
    # (strong foreshortening + any tiny non-planarity / print thickness).
    hard_views = [
        image
        for image in images
        if image.tilt_deg >= 40 and max(image.span_x, image.span_y) >= 0.55
    ]
    hard_high_err = [image for image in hard_views if image.mean_error_px > overall_rms]
    if len(hard_views) >= 5 and len(hard_high_err) >= 3:
        gaps.append(
            CoverageGap(
                code="hard_close_tilts",
                severity="high",
                message=(
                    f"{len(hard_high_err)} close+steeply-tilted views exceed the overall RMS. "
                    "These often drive error even when the board feels flat. Prefer moderate "
                    "tilts (~20–35°) for close shots, and save extreme tilts for mid/far distance."
                ),
            )
        )

    if high_error_fraction > 0.15:
        names = ", ".join(image.name for image in sorted(high_error, key=lambda item: -item.mean_error_px)[:5])
        gaps.append(
            CoverageGap(
                code="outlier_views",
                severity="high",
                message=(
                    f"{len(high_error)}/{n} views have high mean error. "
                    f"Worst: {names}. Remove or retake these and re-calibrate."
                ),
            )
        )

    # Ideal-ish checklist scores for the report summary.
    summary = {
        "view_count": n,
        "mean_of_mean_errors": float(np.mean(mean_errors)),
        "median_mean_error": float(np.median(mean_errors)),
        "max_mean_error": float(np.max(mean_errors)),
        "quadrant_counts": quadrant_counts,
        "tilt_bins": {"low_<15": tilt_low, "med_15_30": tilt_med, "high_>=30": tilt_high},
        "distance_min": float(dist_min),
        "distance_max": float(dist_max),
        "distance_ratio": float(dist_ratio),
        "spatial_empty_cells": empty_cells,
        "spatial_total_cells": total_cells,
        "spatial_empty_corner_cells": empty_corners,
        "edge_centers": {
            "left": near_left,
            "right": near_right,
            "top": near_top,
            "bottom": near_bottom,
        },
        "high_error_views": len(high_error),
    }
    return gaps, summary


def diagnose_detections(
    detections: DetectionSet,
    calibration: CalibrationResult,
    grid_size: int = 12,
) -> DiagnosisReport:
    """Score collected views against saved intrinsics for coverage / outliers."""
    camera_matrix = np.asarray(calibration.camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(calibration.distortion_coefficients, dtype=np.float64)
    image_size = detections.image_size
    diagnoses: list[ImageDiagnosis] = []
    failed: list[str] = list(detections.failed_images)
    spatial = np.zeros((grid_size, grid_size), dtype=np.int32)

    for view in detections.views:
        ok, rvec, tvec = cv2.solvePnP(
            view.object_points,
            view.image_points,
            camera_matrix,
            dist_coeffs,
        )
        if not ok:
            failed.append(view.name)
            continue

        projected, _ = cv2.projectPoints(
            view.object_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        residuals = view.image_points.reshape(-1, 2) - projected.reshape(-1, 2)
        norms = np.linalg.norm(residuals, axis=1)

        points = view.image_points.reshape(-1, 2)
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        center = (min_xy + max_xy) / 2.0
        width, height = image_size
        span = max_xy - min_xy
        tilt_deg, roll_deg = _rotation_tilt_roll(rvec)
        distance = float(np.linalg.norm(tvec))

        center_x = float(center[0] / width)
        center_y = float(center[1] / height)
        diagnoses.append(
            ImageDiagnosis(
                name=view.name,
                mean_error_px=float(norms.mean()),
                max_error_px=float(norms.max()),
                center_x=center_x,
                center_y=center_y,
                span_x=float(span[0] / width),
                span_y=float(span[1] / height),
                distance=distance,
                tilt_deg=tilt_deg,
                roll_deg=roll_deg,
                quadrant=_quadrant(center_x, center_y),
                corner_count=int(len(points)),
            )
        )
        _accumulate_spatial_grid(view.image_points, image_size, spatial)

    if not diagnoses:
        raise RuntimeError(f"No diagnosable views. Failed: {failed}")

    gaps, summary = _assess_gaps(diagnoses, spatial, calibration.rms_reprojection_error)
    return DiagnosisReport(
        image_size=image_size,
        pattern_size=detections.pattern_size,
        square_size=detections.square_size,
        distortion_model=calibration.distortion_model,
        overall_rms=calibration.rms_reprojection_error,
        images=sorted(diagnoses, key=lambda item: -item.mean_error_px),
        failed_images=failed,
        spatial_grid=spatial.tolist(),
        grid_rows=grid_size,
        grid_cols=grid_size,
        gaps=gaps,
        summary=summary,
    )


def diagnose_calibration(
    folder: Path,
    calibration: CalibrationResult,
    detect_scale: float = 0.35,
    grid_size: int = 12,
    board: str | None = None,
    squares_x: int | None = None,
    squares_y: int | None = None,
    square_size: float | None = None,
    marker_proportion: float | None = None,
    dictionary: str | None = None,
    min_charuco_corners: int = 6,
) -> DiagnosisReport:
    """
    Analyze pose/location coverage and per-image reprojection error.

    Re-detects corners in `folder` using the saved board metadata when present.
    """
    from .calibrate import collect_board_detections
    from .result import BOARD_CHECKERBOARD

    board_type = board or calibration.board_type or BOARD_CHECKERBOARD
    pattern = tuple(calibration.pattern_size)
    size = float(square_size if square_size is not None else calibration.square_size)

    resolved_x = squares_x if squares_x is not None else (pattern[0] if pattern[0] else None)
    resolved_y = squares_y if squares_y is not None else (pattern[1] if pattern[1] else None)
    if resolved_x is None or resolved_y is None:
        raise ValueError(
            "Diagnose needs --squares-x/--squares-y "
            "(or a JSON file that stores pattern_size)."
        )

    detections = collect_board_detections(
        folder=folder,
        board=board_type,
        squares_x=resolved_x,
        squares_y=resolved_y,
        square_size=size,
        detect_scale=detect_scale,
        marker_proportion=(
            marker_proportion
            if marker_proportion is not None
            else (calibration.marker_proportion or 0.7)
        ),
        dictionary=dictionary or calibration.dictionary or "DICT_4X4_50",
        min_charuco_corners=min_charuco_corners,
        min_views=1,
    )
    return diagnose_detections(detections, calibration, grid_size=grid_size)


def render_diagnosis_image(report: DiagnosisReport, path: Path) -> Path:
    """
    Write a multi-panel PNG summarizing coverage and per-view error.

    Panels:
      1) board centers / spans on the image frame (color = mean error)
      2) corner-hit density heatmap
      3) tilt vs distance (color = mean error)
      4) worst views by mean reprojection error
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    images = report.images
    if not images:
        raise RuntimeError("Diagnosis report has no images to visualize")

    width, height = report.image_size
    errors = np.array([image.mean_error_px for image in images], dtype=np.float64)
    error_vmax = float(max(np.percentile(errors, 90), report.overall_rms, 1.0))

    figure, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    figure.suptitle(
        f"Calibration diagnosis  ·  RMS {report.overall_rms:.2f} px  ·  "
        f"{len(images)} views  ·  {width}×{height}",
        fontsize=13,
    )

    # --- Panel 1: board placement on the frame ---
    axis = axes[0, 0]
    axis.set_xlim(0, 1)
    axis.set_ylim(1, 0)  # image coords: y down
    axis.set_aspect(height / width)
    axis.add_patch(
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="0.4", linewidth=1.2)
    )
    axis.axhline(0.3, color="0.85", linewidth=0.6, linestyle="--")
    axis.axhline(0.7, color="0.85", linewidth=0.6, linestyle="--")
    axis.axvline(0.3, color="0.85", linewidth=0.6, linestyle="--")
    axis.axvline(0.7, color="0.85", linewidth=0.6, linestyle="--")

    for image in images:
        left = image.center_x - image.span_x / 2.0
        top = image.center_y - image.span_y / 2.0
        axis.add_patch(
            Rectangle(
                (left, top),
                image.span_x,
                image.span_y,
                fill=False,
                edgecolor="0.75",
                linewidth=0.6,
                alpha=0.7,
            )
        )

    scatter = axis.scatter(
        [image.center_x for image in images],
        [image.center_y for image in images],
        c=errors,
        s=[40 + 180 * max(image.span_x, image.span_y) for image in images],
        cmap="magma",
        vmin=0.0,
        vmax=error_vmax,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    axis.set_title("Board centers in the frame (dot size ≈ board span)")
    axis.set_xlabel("Normalized x (left → right)")
    axis.set_ylabel("Normalized y (top → bottom)")
    figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04, label="Mean error (px)")

    # --- Panel 2: corner density ---
    axis = axes[0, 1]
    grid = np.asarray(report.spatial_grid, dtype=np.float64)
    heatmap = axis.imshow(
        grid,
        cmap="viridis",
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_title(f"Corner hit density ({report.grid_rows}×{report.grid_cols} bins)")
    axis.set_xlabel("Image x bins")
    axis.set_ylabel("Image y bins")
    figure.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04, label="Corner count")

    # --- Panel 3: tilt vs distance ---
    axis = axes[1, 0]
    scatter = axis.scatter(
        [image.distance for image in images],
        [image.tilt_deg for image in images],
        c=errors,
        s=[30 + 120 * max(image.span_x, image.span_y) for image in images],
        cmap="magma",
        vmin=0.0,
        vmax=error_vmax,
        edgecolors="white",
        linewidths=0.4,
    )
    axis.axhline(30, color="0.6", linestyle="--", linewidth=0.8, label="30° tilt")
    axis.set_title("Pose mix: tilt vs distance")
    axis.set_xlabel(f"Distance (square-size units, square={report.square_size})")
    axis.set_ylabel("Board tilt (degrees)")
    axis.legend(loc="best", fontsize=8)
    figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04, label="Mean error (px)")

    # --- Panel 4: worst views ---
    axis = axes[1, 1]
    worst = images[: min(12, len(images))]
    labels = [image.name.replace(".MP.jpg", "").replace(".jpg", "")[-15:] for image in worst]
    values = [image.mean_error_px for image in worst]
    colors = plt.cm.magma(np.clip(np.array(values) / error_vmax, 0.0, 1.0))
    axis.barh(range(len(worst))[::-1], values[::-1], color=colors[::-1])
    axis.set_yticks(range(len(worst)))
    axis.set_yticklabels(labels[::-1], fontsize=8)
    axis.axvline(report.overall_rms, color="crimson", linestyle="--", linewidth=1.0, label="overall RMS")
    axis.set_xlabel("Mean reprojection error (px)")
    axis.set_title("Worst views")
    axis.legend(loc="lower right", fontsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
