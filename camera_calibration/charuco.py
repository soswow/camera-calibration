"""ChArUco board construction and corner collection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco

from .detection import DetectedView, DetectionSet
from .images import (
    DETECTION_ROTATIONS,
    choose_canonical_image_size,
    list_images,
    normalize_to_calibration_size,
    read_calibration_image,
    rotate_for_detection,
    unrotate_detection_points,
)
from .result import BOARD_CHARUCO


def get_dictionary(name: str):
    if not hasattr(aruco, name):
        available = sorted(item for item in dir(aruco) if item.startswith("DICT_"))
        raise ValueError(
            f"Unknown dictionary {name!r}. Available: {', '.join(available)}"
        )
    return aruco.getPredefinedDictionary(getattr(aruco, name))


def create_board(
    squares_x: int,
    squares_y: int,
    square_size: float,
    marker_proportion: float,
    dictionary,
):
    marker_size = square_size * marker_proportion
    if hasattr(aruco, "CharucoBoard"):
        return aruco.CharucoBoard(
            (squares_x, squares_y),
            square_size,
            marker_size,
            dictionary,
        )
    return aruco.CharucoBoard_create(
        squares_x,
        squares_y,
        square_size,
        marker_size,
        dictionary,
    )


def board_chessboard_corners(board) -> np.ndarray:
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype=np.float32)
    return np.asarray(board.chessboardCorners, dtype=np.float32)


def find_charuco_corners(
    gray: np.ndarray,
    board,
    min_corners: int = 6,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Detect interpolated ChArUco chessboard corners.

    Returns (corners, ids) or None when too few corners are found.
    """
    if hasattr(aruco, "CharucoDetector"):
        detector = aruco.CharucoDetector(board)
        corners, ids, _marker_corners, _marker_ids = detector.detectBoard(gray)
    else:
        dictionary = board.getDictionary() if hasattr(board, "getDictionary") else board.dictionary
        marker_corners, marker_ids, _ = aruco.detectMarkers(gray, dictionary)
        if marker_ids is None or len(marker_ids) < 1:
            return None
        _, corners, ids = aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )

    if corners is None or ids is None or len(ids) < min_corners:
        return None
    return corners, ids


def find_charuco_corners_with_detection_rotation(
    image: np.ndarray,
    board,
    min_corners: int = 6,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Detect ChArUco corners with temporary image rotations.

    Returned image points are mapped back to the original calibration image
    frame. Object-point lookup remains based on ChArUco IDs, so board pose in
    the photo does not need to match --squares-x/--squares-y visually.
    """
    height, width = image.shape[:2]
    for rotation in DETECTION_ROTATIONS:
        rotated = rotate_for_detection(image, rotation)
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        detection = find_charuco_corners(gray, board, min_corners=min_corners)
        if detection is None:
            continue

        corners, ids = detection
        return unrotate_detection_points(corners, (width, height), rotation), ids

    return None


def draw_detected_charuco(
    image: np.ndarray,
    corners: np.ndarray,
    ids: np.ndarray,
) -> np.ndarray:
    annotated = image.copy()
    if hasattr(aruco, "drawDetectedCornersCharuco"):
        aruco.drawDetectedCornersCharuco(annotated, corners, ids)
    else:
        cv2.drawChessboardCorners(annotated, (len(corners), 1), corners, True)
    return annotated


def collect_detections(
    folder: Path,
    squares_x: int,
    squares_y: int,
    square_size: float,
    marker_proportion: float,
    dictionary_name: str,
    min_corners: int = 6,
    preview_dir: Path | None = None,
    min_views: int = 3,
) -> DetectionSet:
    """Detect ChArUco corners in every image (partial boards are allowed)."""
    images = list_images(folder)
    if not images:
        raise FileNotFoundError(f"No images found in {folder}")

    dictionary = get_dictionary(dictionary_name)
    board = create_board(
        squares_x, squares_y, square_size, marker_proportion, dictionary
    )
    object_corners = board_chessboard_corners(board)
    image_size = choose_canonical_image_size(images)

    views: list[DetectedView] = []
    failed_images: list[str] = []

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

        detection = find_charuco_corners_with_detection_rotation(
            image,
            board,
            min_corners=min_corners,
        )
        if detection is None:
            failed_images.append(image_path.name)
            continue

        corners, ids = detection
        ids_flat = ids.flatten()
        views.append(
            DetectedView(
                name=image_path.name,
                object_points=object_corners[ids_flat],
                image_points=corners.reshape(-1, 1, 2).astype(np.float32),
                was_rotated=(
                    calibration_image.was_transformed or was_size_normalized
                ),
            )
        )

        if preview_dir is not None:
            annotated = draw_detected_charuco(image, corners, ids)
            cv2.imwrite(str(preview_dir / image_path.name), annotated)

    if len(views) < min_views:
        raise RuntimeError(
            f"Need at least {min_views} successful ChArUco detections, got {len(views)}. "
            f"Failed: {failed_images}. Check --squares-x/--squares-y, "
            "--dictionary, --marker-proportion, and that markers are readable. "
            "Calibration uses inverse EXIF orientation to keep a consistent "
            "camera pixel frame; remove or separately calibrate images whose "
            "normalized dimensions still differ."
        )

    return DetectionSet(
        image_size=image_size,
        pattern_size=(squares_x, squares_y),
        square_size=square_size,
        views=views,
        failed_images=failed_images,
        board_type=BOARD_CHARUCO,
        dictionary=dictionary_name,
        marker_proportion=marker_proportion,
    )
