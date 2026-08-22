"""Held-out reprojection and straight-line validation for saved intrinsics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .charuco import (
    board_chessboard_corners,
    create_board,
    find_charuco_corners_with_detection_rotation,
    get_dictionary,
)
from .checkerboard import (
    find_corners_with_detection_rotation,
    inner_corners_from_squares,
)
from .detection import DetectedView, object_points_grid
from .images import (
    IMAGE_EXTENSIONS,
    list_images,
    normalize_to_calibration_size,
    read_calibration_image,
)
from .result import BOARD_CHARUCO, BOARD_CHECKERBOARD, CalibrationResult


@dataclass
class ErrorMetrics:
    """Distribution summary for non-negative pixel errors."""

    rms_px: float
    mean_px: float
    median_px: float
    p95_px: float
    max_px: float


@dataclass
class StraightnessMetrics:
    """Orthogonal distances from undistorted grid points to fitted grid lines."""

    rms_px: float
    median_px: float
    p95_px: float
    max_px: float
    row_lines: int
    column_lines: int
    distance_count: int


@dataclass
class ImageValidation:
    name: str
    corner_count: int
    reprojection: ErrorMetrics
    straightness: StraightnessMetrics | None
    used_for_calibration: bool
    auto_select_rejected: bool
    visualization: str | None = None


@dataclass
class _DetectedValidationImage:
    path: Path
    image: np.ndarray
    view: DetectedView


@dataclass
class ValidationReport:
    calibration_file: str
    image_size: tuple[int, int]
    board_type: str
    distortion_model: str
    calibration_training_rms_px: float
    validation_reprojection: ErrorMetrics
    validation_straightness: StraightnessMetrics | None
    images: list[ImageValidation]
    failed_images: dict[str, str] = field(default_factory=dict)
    overlap_images: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def _error_metrics(errors: np.ndarray) -> ErrorMetrics:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty error array")
    return ErrorMetrics(
        rms_px=float(np.sqrt(np.mean(np.square(values)))),
        mean_px=float(np.mean(values)),
        median_px=float(np.median(values)),
        p95_px=float(np.percentile(values, 95)),
        max_px=float(np.max(values)),
    )


def _line_distances(points: np.ndarray) -> np.ndarray:
    """Orthogonal point-to-line distances for a least-squares 2D line."""
    center = points.mean(axis=0)
    _u, _s, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    return np.abs((points - center) @ normal)


def _grid_line_groups(
    object_points: np.ndarray,
    image_points: np.ndarray,
    *,
    min_points_per_line: int,
):
    """Yield (row_or_column, points) for usable grid lines."""
    object_xy = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)[:, :2]
    image_xy = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    for label, axis in (("row", 1), ("column", 0)):
        coordinates = object_xy[:, axis]
        for coordinate in np.unique(coordinates):
            line_points = image_xy[np.isclose(coordinates, coordinate)]
            if len(line_points) >= min_points_per_line:
                yield label, line_points


def straightness_errors(
    object_points: np.ndarray,
    undistorted_points: np.ndarray,
    *,
    min_points_per_line: int = 4,
) -> tuple[np.ndarray, int, int]:
    """Return point-to-line errors for all usable board rows and columns."""
    distances: list[np.ndarray] = []
    row_lines = 0
    column_lines = 0
    for label, line_points in _grid_line_groups(
        object_points,
        undistorted_points,
        min_points_per_line=min_points_per_line,
    ):
        distances.append(_line_distances(line_points))
        if label == "row":
            row_lines += 1
        else:
            column_lines += 1

    if not distances:
        return np.empty(0, dtype=np.float64), row_lines, column_lines
    return np.concatenate(distances), row_lines, column_lines


def _fitted_line_segment(points: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """Fit a total-least-squares line and bound it by the observed point range."""
    center = points.mean(axis=0)
    _u, _s, vh = np.linalg.svd(points - center, full_matrices=False)
    direction = vh[0]
    positions = (points - center) @ direction
    endpoints = center + np.outer([positions.min(), positions.max()], direction)
    first = tuple(np.rint(endpoints[0]).astype(int))
    last = tuple(np.rint(endpoints[1]).astype(int))
    return first, last


def render_validation_overlay(
    image: np.ndarray,
    view: DetectedView,
    calibration: CalibrationResult,
    result: ImageValidation,
    destination: Path,
    *,
    alpha: float = 0.0,
    line_opacity: float = 0.55,
) -> Path:
    """Write an undistorted image with translucent fitted grid lines."""
    camera_matrix = np.asarray(calibration.camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(calibration.distortion_coefficients, dtype=np.float64)
    width, height = calibration.image_size
    new_camera_matrix, _roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (width, height),
        alpha,
        (width, height),
    )
    undistorted_image = cv2.undistort(
        image,
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
    )
    undistorted_points = cv2.undistortPoints(
        np.asarray(view.image_points, dtype=np.float64).reshape(-1, 1, 2),
        camera_matrix,
        dist_coeffs,
        P=new_camera_matrix,
    ).reshape(-1, 2)

    line_layer = undistorted_image.copy()
    thickness = max(2, int(round(max(width, height) / 1000.0)))
    colors = {
        "row": (0, 180, 255),      # amber in BGR
        "column": (255, 170, 0),  # blue/cyan in BGR
    }
    for label, line_points in _grid_line_groups(
        view.object_points,
        undistorted_points,
        min_points_per_line=4,
    ):
        first, last = _fitted_line_segment(line_points)
        cv2.line(
            line_layer,
            first,
            last,
            colors[label],
            thickness,
            cv2.LINE_AA,
        )

    rendered = cv2.addWeighted(
        line_layer,
        line_opacity,
        undistorted_image,
        1.0 - line_opacity,
        0.0,
    )
    radius = max(3, thickness + 1)
    for point in undistorted_points:
        center = tuple(np.rint(point).astype(int))
        cv2.circle(rendered, center, radius + 1, (25, 25, 25), -1, cv2.LINE_AA)
        cv2.circle(rendered, center, radius, (245, 245, 245), -1, cv2.LINE_AA)

    font_scale = max(0.65, max(width, height) / 2400.0)
    text_thickness = max(1, int(round(font_scale * 1.5)))
    straight_text = "straightness unavailable"
    if result.straightness is not None:
        straight_text = f"straightness RMS {result.straightness.rms_px:.3f}px"
    text_lines = [
        f"validation RMS {result.reprojection.rms_px:.3f}px  |  " + straight_text,
        "amber: fitted rows  |  blue: fitted columns  |  white: detected intersections",
    ]
    if result.used_for_calibration or result.auto_select_rejected:
        text_lines.append("WARNING: image is not held out")

    margin = max(14, int(round(max(width, height) / 180.0)))
    line_height = int(round(34 * font_scale))
    sizes = [
        cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness,
        )[0]
        for text in text_lines
    ]
    box_width = max(size[0] for size in sizes) + margin * 2
    box_height = line_height * len(text_lines) + margin * 2
    text_layer = rendered.copy()
    cv2.rectangle(text_layer, (0, 0), (box_width, box_height), (0, 0, 0), -1)
    rendered = cv2.addWeighted(text_layer, 0.70, rendered, 0.30, 0.0)
    for index, text in enumerate(text_lines):
        color = (80, 80, 255) if text.startswith("WARNING") else (245, 245, 245)
        baseline_y = margin + line_height * (index + 1) - int(line_height * 0.2)
        cv2.putText(
            rendered,
            text,
            (margin, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            text_thickness,
            cv2.LINE_AA,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), rendered):
        raise RuntimeError(f"Could not write validation visualization: {destination}")
    return destination


def validate_view(
    view: DetectedView,
    calibration: CalibrationResult,
) -> tuple[ImageValidation, np.ndarray, np.ndarray]:
    """Validate one detected view while keeping its saved intrinsics fixed."""
    camera_matrix = np.asarray(calibration.camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(calibration.distortion_coefficients, dtype=np.float64)
    object_points = np.asarray(view.object_points, dtype=np.float64)
    image_points = np.asarray(view.image_points, dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
    )
    if not ok:
        raise RuntimeError("pose estimation failed")

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    residual_vectors = image_points.reshape(-1, 2) - projected.reshape(-1, 2)
    reprojection_errors = np.linalg.norm(residual_vectors, axis=1)

    undistorted = cv2.undistortPoints(
        image_points.reshape(-1, 1, 2),
        camera_matrix,
        dist_coeffs,
        P=camera_matrix,
    ).reshape(-1, 2)
    line_errors, row_lines, column_lines = straightness_errors(
        object_points,
        undistorted,
    )
    straightness = None
    if line_errors.size:
        line_summary = _error_metrics(line_errors)
        straightness = StraightnessMetrics(
            rms_px=line_summary.rms_px,
            median_px=line_summary.median_px,
            p95_px=line_summary.p95_px,
            max_px=line_summary.max_px,
            row_lines=row_lines,
            column_lines=column_lines,
            distance_count=int(line_errors.size),
        )

    used_names = set(calibration.used_images)
    rejected_names = set(calibration.auto_select_rejected or [])
    result = ImageValidation(
        name=view.name,
        corner_count=int(len(reprojection_errors)),
        reprojection=_error_metrics(reprojection_errors),
        straightness=straightness,
        used_for_calibration=view.name in used_names,
        auto_select_rejected=view.name in rejected_names,
    )
    return result, reprojection_errors, line_errors


def _resolve_images(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file: {source}")
        return [source]
    if source.is_dir():
        images = list_images(source)
        if not images:
            raise FileNotFoundError(f"No images found in {source}")
        return images
    raise FileNotFoundError(f"Image or folder not found: {source}")


def _detect_views(
    source: Path,
    calibration: CalibrationResult,
    *,
    board_type: str,
    squares_x: int,
    squares_y: int,
    square_size: float,
    marker_proportion: float,
    dictionary: str,
    min_charuco_corners: int,
    detect_scale: float,
) -> tuple[list[_DetectedValidationImage], dict[str, str]]:
    images = _resolve_images(source)
    detected_images: list[_DetectedValidationImage] = []
    failed: dict[str, str] = {}

    charuco_board = None
    charuco_object_points = None
    checkerboard_inner = None
    if board_type == BOARD_CHARUCO:
        charuco_board = create_board(
            squares_x,
            squares_y,
            square_size,
            marker_proportion,
            get_dictionary(dictionary),
        )
        charuco_object_points = board_chessboard_corners(charuco_board)
    elif board_type == BOARD_CHECKERBOARD:
        checkerboard_inner = inner_corners_from_squares(squares_x, squares_y)
    else:
        raise ValueError(f"Unknown board type {board_type!r}")

    for image_path in images:
        loaded = read_calibration_image(image_path)
        if loaded is None:
            failed[image_path.name] = "image could not be read"
            continue
        sized = normalize_to_calibration_size(loaded.image, calibration.image_size)
        if sized is None:
            height, width = loaded.image.shape[:2]
            expected_width, expected_height = calibration.image_size
            failed[image_path.name] = (
                f"image size {width}x{height} does not match calibration "
                f"{expected_width}x{expected_height}"
            )
            continue
        image, was_size_normalized = sized

        if board_type == BOARD_CHARUCO:
            assert charuco_board is not None and charuco_object_points is not None
            detection = find_charuco_corners_with_detection_rotation(
                image,
                charuco_board,
                min_corners=min_charuco_corners,
            )
            if detection is None:
                failed[image_path.name] = "ChArUco board was not detected"
                continue
            corners, ids = detection
            object_points = charuco_object_points[ids.flatten()]
        else:
            assert checkerboard_inner is not None
            detection = find_corners_with_detection_rotation(
                image,
                checkerboard_inner,
                detect_scale=detect_scale,
            )
            if detection is None:
                failed[image_path.name] = "checkerboard was not detected"
                continue
            corners, used_inner = detection
            object_points = object_points_grid(*used_inner, square_size)

        view = DetectedView(
            name=image_path.name,
            object_points=np.asarray(object_points, dtype=np.float32),
            image_points=np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2),
            was_rotated=loaded.was_transformed or was_size_normalized,
        )
        detected_images.append(
            _DetectedValidationImage(path=image_path, image=image, view=view)
        )

    return detected_images, failed


def _visualization_destination(
    output: Path,
    *,
    source_is_file: bool,
    image_path: Path,
) -> Path:
    if source_is_file and output.suffix.lower() in IMAGE_EXTENSIONS:
        return output
    return output / f"{image_path.stem}-validation.png"


def validate_calibration(
    source: Path,
    calibration: CalibrationResult,
    *,
    calibration_file: Path | None = None,
    board_type: str | None = None,
    squares_x: int | None = None,
    squares_y: int | None = None,
    square_size: float | None = None,
    marker_proportion: float | None = None,
    dictionary: str | None = None,
    min_charuco_corners: int = 6,
    detect_scale: float = 0.35,
    visualization_output: Path | None = None,
    visualization_alpha: float = 0.0,
    line_opacity: float = 0.55,
) -> ValidationReport:
    """Detect validation boards and score them against fixed saved intrinsics."""
    resolved_board = board_type or calibration.board_type
    resolved_x = squares_x if squares_x is not None else calibration.pattern_size[0]
    resolved_y = squares_y if squares_y is not None else calibration.pattern_size[1]
    resolved_square = square_size if square_size is not None else calibration.square_size
    resolved_marker = (
        marker_proportion
        if marker_proportion is not None
        else (calibration.marker_proportion or 0.7)
    )
    resolved_dictionary = dictionary or calibration.dictionary or "DICT_4X4_50"

    if resolved_x < 2 or resolved_y < 2:
        raise ValueError(
            "Board dimensions are missing; provide --squares-x and --squares-y"
        )
    if resolved_square <= 0:
        raise ValueError("Square size is missing; provide --square-size")
    if not (0.0 < resolved_marker < 1.0):
        raise ValueError("--marker-proportion must be in (0, 1)")

    detected_images, failed = _detect_views(
        source,
        calibration,
        board_type=resolved_board,
        squares_x=resolved_x,
        squares_y=resolved_y,
        square_size=resolved_square,
        marker_proportion=resolved_marker,
        dictionary=resolved_dictionary,
        min_charuco_corners=min_charuco_corners,
        detect_scale=detect_scale,
    )
    if not detected_images:
        details = "; ".join(f"{name}: {reason}" for name, reason in failed.items())
        raise RuntimeError(f"No validation boards detected. {details}")

    image_results: list[ImageValidation] = []
    all_reprojection: list[np.ndarray] = []
    all_straightness: list[np.ndarray] = []
    row_lines = 0
    column_lines = 0
    for detected in detected_images:
        view = detected.view
        try:
            result, reprojection_errors, line_errors = validate_view(view, calibration)
        except RuntimeError as error:
            failed[view.name] = str(error)
            continue
        image_results.append(result)
        all_reprojection.append(reprojection_errors)
        if line_errors.size:
            all_straightness.append(line_errors)
            assert result.straightness is not None
            row_lines += result.straightness.row_lines
            column_lines += result.straightness.column_lines
        if visualization_output is not None:
            destination = _visualization_destination(
                visualization_output,
                source_is_file=source.is_file(),
                image_path=detected.path,
            )
            result.visualization = str(
                render_validation_overlay(
                    detected.image,
                    view,
                    calibration,
                    result,
                    destination,
                    alpha=visualization_alpha,
                    line_opacity=line_opacity,
                )
            )

    if not image_results:
        raise RuntimeError("Pose estimation failed for every detected validation image")

    reprojection_values = np.concatenate(all_reprojection)
    straightness = None
    if all_straightness:
        straightness_values = np.concatenate(all_straightness)
        summary = _error_metrics(straightness_values)
        straightness = StraightnessMetrics(
            rms_px=summary.rms_px,
            median_px=summary.median_px,
            p95_px=summary.p95_px,
            max_px=summary.max_px,
            row_lines=row_lines,
            column_lines=column_lines,
            distance_count=int(straightness_values.size),
        )

    overlap = sorted(
        result.name
        for result in image_results
        if result.used_for_calibration or result.auto_select_rejected
    )
    return ValidationReport(
        calibration_file=str(calibration_file) if calibration_file else "",
        image_size=calibration.image_size,
        board_type=resolved_board,
        distortion_model=calibration.distortion_model,
        calibration_training_rms_px=calibration.rms_reprojection_error,
        validation_reprojection=_error_metrics(reprojection_values),
        validation_straightness=straightness,
        images=image_results,
        failed_images=failed,
        overlap_images=overlap,
    )
