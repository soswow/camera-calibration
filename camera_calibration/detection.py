"""Shared detection set + OpenCV calibrateCamera fit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .result import BOARD_CHECKERBOARD, CalibrationResult

DISTORTION_MODELS = {
    "full": 0,
    "simple": cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
    "k1": cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
}


@dataclass
class DetectedView:
    """One successfully detected board view."""

    name: str
    object_points: np.ndarray
    image_points: np.ndarray
    was_rotated: bool


@dataclass
class DetectionSet:
    """All detections collected from a folder, ready for calibrateCamera."""

    image_size: tuple[int, int]
    pattern_size: tuple[int, int]  # square counts (squares_x, squares_y)
    square_size: float
    views: list[DetectedView]
    failed_images: list[str]
    board_type: str = BOARD_CHECKERBOARD
    dictionary: str | None = None
    marker_proportion: float | None = None


def object_points_grid(cols: int, rows: int, square_size: float) -> np.ndarray:
    """3D points of a checkerboard in the board's local coordinate frame."""
    points = np.zeros((cols * rows, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[:, :2] = grid * square_size
    return points


def fit_intrinsics(
    detections: DetectionSet,
    distortion_model: str = "simple",
) -> CalibrationResult:
    """Run cv2.calibrateCamera on an already-collected DetectionSet."""
    if distortion_model not in DISTORTION_MODELS:
        raise ValueError(
            f"Unknown distortion_model {distortion_model!r}. "
            f"Choose from: {', '.join(DISTORTION_MODELS)}"
        )
    if len(detections.views) < 3:
        raise RuntimeError(
            f"Need at least 3 views to calibrate, got {len(detections.views)}"
        )

    flags = DISTORTION_MODELS[distortion_model]
    object_points = [view.object_points for view in detections.views]
    image_points = [view.image_points for view in detections.views]
    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        detections.image_size,
        None,
        None,
        flags=flags,
    )

    from .fitzgibbon import estimate_fitzgibbon_lambda

    fitz = estimate_fitzgibbon_lambda(detections, camera_matrix)

    return CalibrationResult(
        image_size=detections.image_size,
        camera_matrix=camera_matrix.tolist(),
        distortion_coefficients=dist_coeffs.ravel().tolist(),
        rms_reprojection_error=float(rms),
        used_images=[view.name for view in detections.views],
        failed_images=list(detections.failed_images),
        pattern_size=detections.pattern_size,
        square_size=detections.square_size,
        distortion_model=distortion_model,
        board_type=detections.board_type,
        dictionary=detections.dictionary,
        marker_proportion=detections.marker_proportion,
        rotated_images=[view.name for view in detections.views if view.was_rotated],
        fitzgibbon_lambda=fitz.lambda_,
        fitzgibbon_rms_reprojection_error=fitz.rms_reprojection_error,
    )


def calibrate_detections(
    detections: DetectionSet,
    distortion_model: str = "simple",
    auto_select: bool = False,
    auto_select_max_keep: int | None = None,
    auto_select_error_factor: float = 1.5,
    auto_select_error_floor: float = 2.0,
) -> CalibrationResult:
    """Fit intrinsics, optionally dropping outliers while keeping pose diversity."""
    if not auto_select:
        return fit_intrinsics(detections, distortion_model=distortion_model)

    from .auto_select import auto_select_and_refit

    result, _selection = auto_select_and_refit(
        detections,
        distortion_model=distortion_model,
        error_factor=auto_select_error_factor,
        error_floor_px=auto_select_error_floor,
        max_keep=auto_select_max_keep,
    )
    return result
