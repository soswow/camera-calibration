"""Undistort images using saved camera calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .images import (
    IMAGE_EXTENSIONS,
    list_images,
    normalize_to_calibration_size,
    read_calibration_image,
)
from .result import CalibrationResult


def load_calibration(path: Path) -> CalibrationResult:
    """Load project JSON or ROS camera_info YAML."""
    return CalibrationResult.from_path(path)


def undistort_image(
    image: np.ndarray,
    calibration: CalibrationResult,
    alpha: float = 0.0,
) -> np.ndarray:
    """
    Remove lens distortion from a BGR image.

    alpha=0 crops to the largest valid rectangle (no black borders).
    alpha=1 keeps all source pixels (may introduce black borders).
    """
    height, width = image.shape[:2]
    calibrated_width, calibrated_height = calibration.image_size
    if (width, height) != (calibrated_width, calibrated_height):
        raise ValueError(
            f"Image size {width}x{height} does not match calibration "
            f"{calibrated_width}x{calibrated_height}"
        )

    camera_matrix = np.asarray(calibration.camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(calibration.distortion_coefficients, dtype=np.float64)
    new_camera_matrix, _roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (width, height),
        alpha,
        (width, height),
    )
    return cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)


@dataclass
class UndistortBatchResult:
    """Summary of undistorting one or more images."""

    written: list[str]
    failed: list[str]


def _resolve_jobs(source: Path, output: Path) -> list[tuple[Path, Path]]:
    """Map each source image to its destination path."""
    if source.is_file():
        if output.suffix.lower() in IMAGE_EXTENSIONS:
            return [(source, output)]
        output.mkdir(parents=True, exist_ok=True)
        return [(source, output / source.name)]

    if source.is_dir():
        images = list_images(source)
        if not images:
            raise FileNotFoundError(f"No images found in {source}")
        output.mkdir(parents=True, exist_ok=True)
        return [(image_path, output / image_path.name) for image_path in images]

    raise FileNotFoundError(f"Not a file or directory: {source}")


def undistort_path(
    source: Path,
    calibration: CalibrationResult,
    output: Path,
    alpha: float = 0.0,
) -> UndistortBatchResult:
    """
    Undistort a single image or every image in a folder.

    For a file, `output` may be a destination file or a folder.
    For a folder, `output` is the destination directory (same filenames).
    """
    jobs = _resolve_jobs(source, output)
    written: list[str] = []
    failed: list[str] = []

    for image_path, destination in jobs:
        calibration_image = read_calibration_image(image_path)
        if calibration_image is None:
            failed.append(image_path.name)
            continue

        try:
            sized = normalize_to_calibration_size(
                calibration_image.image,
                calibration.image_size,
            )
            if sized is None:
                raise ValueError
            image, _was_size_normalized = sized
            undistorted = undistort_image(image, calibration, alpha=alpha)
        except ValueError:
            failed.append(image_path.name)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), undistorted):
            failed.append(image_path.name)
            continue
        written.append(str(destination))

    if not written:
        raise RuntimeError(
            f"No images were undistorted. Failed: {failed}. "
            "Check that image resolution matches the calibration."
        )

    return UndistortBatchResult(written=written, failed=failed)
