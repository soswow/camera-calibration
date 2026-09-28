"""Single-camera observations and Brown/angular calibration backends."""

from __future__ import annotations

from dataclasses import dataclass

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
    source_sha256: str | None = None
    corner_ids: list[int] | None = None
    detection_details: dict | None = None


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
    pixel_policy: str = "encoded"
    board_legacy_pattern: bool = False
    board_first_marker_id: int = 0


def object_points_grid(cols: int, rows: int, square_size: float) -> np.ndarray:
    """3D points of a checkerboard in the board's local coordinate frame."""
    points = np.zeros((cols * rows, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[:, :2] = grid * square_size
    return points


def fit_intrinsics(
    detections: DetectionSet,
    distortion_model: str = "simple",
    **fit_options,
) -> CalibrationResult:
    """Run cv2.calibrateCamera on an already-collected DetectionSet."""
    from .models import ANGULAR_MODELS, CameraModel

    if distortion_model in ANGULAR_MODELS:
        from .observations import partition

        detections, _, _ = partition(detections, every=0)
        from .fitting import fit_angular

        intr, poses, checks = fit_angular(
            detections.views,
            detections.image_size,
            asymmetric=distortion_model == "angular-asymmetric",
            **fit_options,
        )
        terms = fit_options.get("terms", 2)
        matrix = [[intr[0], 0, intr[2]], [0, intr[1], intr[3]], [0, 0, 1]]
        asym = (
            intr[-2:].tolist()
            if distortion_model == "angular-asymmetric"
            else [0.0, 0.0]
        )
        model = CameraModel(
            matrix,
            intr[4 : 4 + terms],
            distortion_model,
            asym,
            fit_options.get("max_angle", 110.0),
        )
        errors = np.concatenate(
            [
                (
                    model.project(v.object_points, p) - v.image_points.reshape(-1, 2)
                ).ravel()
                for v, p in zip(detections.views, poses)
            ]
        )
        return CalibrationResult(
            image_size=detections.image_size,
            camera_matrix=matrix,
            distortion_coefficients=intr[4 : 4 + terms].tolist(),
            asymmetric_coefficients=tuple(asym),
            max_angle_deg=model.max_angle_deg,
            rms_reprojection_error=float(np.sqrt(np.mean(errors**2) * 2)),
            used_images=[v.name for v in detections.views],
            failed_images=list(detections.failed_images),
            pattern_size=detections.pattern_size,
            square_size=detections.square_size,
            distortion_model=distortion_model,
            board_type=detections.board_type,
            dictionary=detections.dictionary,
            marker_proportion=detections.marker_proportion,
            pixel_policy=detections.pixel_policy,
            optimizer=checks,
            board_legacy_pattern=detections.board_legacy_pattern,
            board_first_marker_id=detections.board_first_marker_id,
        )
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
    object_points = [
        np.asarray(view.object_points, dtype=np.float32).reshape(-1, 3)
        for view in detections.views
    ]
    image_points = [
        np.asarray(view.image_points, dtype=np.float32).reshape(-1, 1, 2)
        for view in detections.views
    ]
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
        pixel_policy=detections.pixel_policy,
        board_legacy_pattern=detections.board_legacy_pattern,
        board_first_marker_id=detections.board_first_marker_id,
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
    **fit_options,
) -> CalibrationResult:
    """Fit intrinsics, optionally dropping outliers while keeping pose diversity."""
    if not auto_select:
        return fit_intrinsics(
            detections, distortion_model=distortion_model, **fit_options
        )

    from .auto_select import auto_select_and_refit

    result, _selection = auto_select_and_refit(
        detections,
        distortion_model=distortion_model,
        error_factor=auto_select_error_factor,
        error_floor_px=auto_select_error_floor,
        max_keep=auto_select_max_keep,
        fit_options=fit_options,
    )
    return result
