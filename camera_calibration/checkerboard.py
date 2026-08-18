"""Checkerboard corner detection and folder collection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .detection import DetectedView, DetectionSet, object_points_grid
from .images import (
    DETECTION_ROTATIONS,
    choose_canonical_image_size,
    list_images,
    normalize_to_calibration_size,
    read_calibration_image,
    rotate_for_detection,
    unrotate_detection_points,
)
from .result import BOARD_CHECKERBOARD


def inner_corners_from_squares(squares_x: int, squares_y: int) -> tuple[int, int]:
    """OpenCV checkerboard size is inner corners: squares minus one on each axis."""
    if squares_x < 3 or squares_y < 3:
        raise ValueError(
            "Checkerboard needs at least 3 squares on each side "
            f"(got {squares_x}x{squares_y})"
        )
    return squares_x - 1, squares_y - 1


def squares_from_inner_corners(cols: int, rows: int) -> tuple[int, int]:
    return cols + 1, rows + 1


def find_corners(
    gray: np.ndarray,
    inner_corners: tuple[int, int],
    detect_scale: float = 0.35,
    allow_swapped: bool = True,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    """
    Detect and refine checkerboard inner corners.

    Large photos are tried at a downscale first, then fallback scales.
    Returns (corners, inner_corners_used) or None if not found.
    """
    cols, rows = inner_corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    scales: list[float] = []
    if max(gray.shape) > 1600:
        scales.append(detect_scale)
        for candidate_scale in (0.5, 0.25, 1.0):
            if candidate_scale not in scales:
                scales.append(candidate_scale)
    else:
        scales.append(1.0)

    candidates = [(cols, rows)]
    if allow_swapped and cols != rows:
        candidates.append((rows, cols))

    found_corners: np.ndarray | None = None
    used_size = inner_corners
    used_scale = 1.0
    for scale in scales:
        search = gray
        if scale < 1.0:
            search = cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        for candidate in candidates:
            found, corners = cv2.findChessboardCorners(search, candidate, flags)
            if found:
                found_corners = corners
                used_size = candidate
                used_scale = scale
                break
        if found_corners is not None:
            break

    if found_corners is None:
        return None

    if used_scale < 1.0:
        found_corners = found_corners / used_scale

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    refined = cv2.cornerSubPix(gray, found_corners, (11, 11), (-1, -1), criteria)
    return refined, used_size


def find_corners_with_detection_rotation(
    image: np.ndarray,
    inner_corners: tuple[int, int],
    detect_scale: float = 0.35,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    """
    Detect board corners, allowing temporary image rotations.

    Corners are always returned in the original image coordinate frame used for
    calibration. Rotating only the detection image removes an artificial
    "board must be upright" restriction without mixing camera pixel frames.
    """
    height, width = image.shape[:2]
    for rotation in DETECTION_ROTATIONS:
        rotated = rotate_for_detection(image, rotation)
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        detection = find_corners(
            gray,
            inner_corners,
            detect_scale=detect_scale,
            allow_swapped=False,
        )
        if detection is None:
            continue

        corners, used_inner = detection
        return (
            unrotate_detection_points(corners, (width, height), rotation),
            used_inner,
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return find_corners(gray, inner_corners, detect_scale=detect_scale)


def draw_detected_corners(
    image: np.ndarray,
    pattern_size: tuple[int, int],
    corners: np.ndarray,
) -> np.ndarray:
    """Return a copy of image with detected corners drawn on it."""
    annotated = image.copy()
    cv2.drawChessboardCorners(annotated, pattern_size, corners, True)
    return annotated


def collect_detections(
    folder: Path,
    squares_x: int,
    squares_y: int,
    square_size: float,
    detect_scale: float = 0.35,
    preview_dir: Path | None = None,
    min_views: int = 3,
) -> DetectionSet:
    """Detect checkerboard corners in every image."""
    images = list_images(folder)
    if not images:
        raise FileNotFoundError(f"No images found in {folder}")

    requested_inner = inner_corners_from_squares(squares_x, squares_y)
    image_size = choose_canonical_image_size(images)

    views: list[DetectedView] = []
    failed_images: list[str] = []
    detected_squares: tuple[int, int] | None = None

    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        calibration_image = read_calibration_image(image_path)
        if calibration_image is None:
            failed_images.append(image_path.name)
            continue

        sized = normalize_to_calibration_size(calibration_image.image, image_size)
        if sized is None:
            failed_images.append(image_path.name)
            continue
        image, was_size_normalized = sized

        detection = find_corners_with_detection_rotation(
            image,
            requested_inner,
            detect_scale=detect_scale,
        )
        if detection is None:
            failed_images.append(image_path.name)
            continue

        corners, used_inner = detection
        used_squares = squares_from_inner_corners(*used_inner)
        if detected_squares is None:
            detected_squares = used_squares
        elif used_squares != detected_squares:
            failed_images.append(image_path.name)
            continue

        views.append(
            DetectedView(
                name=image_path.name,
                object_points=object_points_grid(*used_inner, square_size),
                image_points=corners,
                was_rotated=(
                    calibration_image.was_transformed or was_size_normalized
                ),
            )
        )

        if preview_dir is not None:
            annotated = draw_detected_corners(image, used_inner, corners)
            cv2.imwrite(str(preview_dir / image_path.name), annotated)

    if len(views) < min_views:
        raise RuntimeError(
            f"Need at least {min_views} successful detections, got {len(views)}. "
            f"Failed: {failed_images}. Check --squares-x/--squares-y "
            "and that the board is fully visible and sharp. Calibration uses "
            "inverse EXIF orientation to keep a consistent camera pixel frame; "
            "remove or separately calibrate images whose normalized dimensions "
            "still differ."
        )

    assert detected_squares is not None
    return DetectionSet(
        image_size=image_size,
        pattern_size=detected_squares,
        square_size=square_size,
        views=views,
        failed_images=failed_images,
        board_type=BOARD_CHECKERBOARD,
    )
