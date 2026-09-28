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
    legacy_pattern=False,
    first_marker_id=0,
):
    from .board import make_board

    return make_board(
        squares_x,
        squares_y,
        square_size,
        square_size * marker_proportion,
        dictionary,
        legacy_pattern=legacy_pattern,
        first_marker_id=first_marker_id,
    )


def board_chessboard_corners(board) -> np.ndarray:
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype=np.float32)
    return np.asarray(board.chessboardCorners, dtype=np.float32)


def find_charuco_corners(
    gray: np.ndarray,
    board,
    min_corners: int = 6,
    details=None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Detect interpolated ChArUco chessboard corners.

    Returns (corners, ids) or None when too few corners are found.
    """
    if details is None:
        details = {}
    attempts = details.setdefault("attempts", [])
    parameters = aruco.CharucoParameters()
    parameters.checkMarkers = True
    detector = aruco.CharucoDetector(board, parameters)
    h, w = gray.shape
    best = None
    for scale in (1.0, 0.75, 0.5, 0.33):
        sw, sh = max(16, round(w * scale)), max(16, round(h * scale))
        search = (
            gray
            if scale == 1
            else cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
        )
        corners, ids, _, _ = detector.detectBoard(search)
        attempt = {
            "scale_xy": [sw / w, sh / h],
            "detected_corners": 0 if ids is None else len(ids),
            "usable": False,
        }
        attempts.append(attempt)
        if corners is None or ids is None:
            continue
        xy = (corners.reshape(-1, 2).astype(float) + 0.5) / np.array(
            [sw / w, sh / h]
        ) - 0.5
        if scale != 1:
            refined = cv2.cornerSubPix(
                gray,
                xy.astype(np.float32).reshape(-1, 1, 2),
                (3, 3),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01),
            ).reshape(-1, 2)
            good = np.isfinite(refined).all(axis=1) & (
                np.linalg.norm(refined - xy, axis=1) <= 2 / min(sw / w, sh / h)
            )
            xy, ids = refined[good], ids[good]
        if len(ids) < min_corners:
            continue
        objects = board_chessboard_corners(board)[ids.ravel()]
        if np.linalg.matrix_rank(objects - objects.mean(axis=0)) < 2:
            continue
        attempt.update(usable=True, retained_corners=len(ids))
        candidate = (xy.astype(np.float32).reshape(-1, 1, 2), ids)
        if scale == 1:
            details["selected_scale_xy"] = [sw / w, sh / h]
            return candidate
        if best is None or len(ids) > len(best[1]):
            best = candidate
            details["selected_scale_xy"] = [sw / w, sh / h]
    return best


def find_charuco_corners_with_detection_rotation(
    image: np.ndarray,
    board,
    min_corners: int = 6,
    details=None,
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
        trial = {"rotation_degrees": rotation}
        detection = find_charuco_corners(
            gray, board, min_corners=min_corners, details=trial
        )
        if details is not None:
            details.setdefault("rotation_attempts", []).append(trial)
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
    pixel_policy="encoded",
    masks=None,
    legacy_pattern=False,
    first_marker_id=0,
) -> DetectionSet:
    """Detect ChArUco corners in every image (partial boards are allowed)."""
    images = list_images(folder)
    if not images:
        raise FileNotFoundError(f"No images found in {folder}")

    dictionary = get_dictionary(dictionary_name)
    board = create_board(
        squares_x,
        squares_y,
        square_size,
        marker_proportion,
        dictionary,
        legacy_pattern,
        first_marker_id,
    )
    object_corners = board_chessboard_corners(board)
    image_size = choose_canonical_image_size(images, pixel_policy)

    views: list[DetectedView] = []
    failed_images: list[str] = []

    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        calibration_image = read_calibration_image(image_path, pixel_policy)
        if calibration_image is None:
            failed_images.append(image_path.name)
            continue

        sized = normalize_to_calibration_size(
            calibration_image.image, image_size, pixel_policy
        )
        if sized is None:
            failed_images.append(image_path.name)
            continue
        image, was_size_normalized = sized

        from .masks import mask_for
        from .observations import image_hash

        excluded = mask_for(image.shape, (masks or {}).get(image_path.name, []))
        search = image.copy()
        search[excluded != 0] = 127
        details = {
            "multiscale_fallback": True,
            "check_markers": True,
            "exclusion_polygons": (masks or {}).get(image_path.name, []),
        }
        detection = find_charuco_corners_with_detection_rotation(
            search,
            board,
            min_corners=min_corners,
            details=details,
        )
        if detection is None:
            failed_images.append(image_path.name)
            continue

        corners, ids = detection
        margin = cv2.dilate(excluded, np.ones((25, 25), np.uint8))
        xy = np.rint(corners.reshape(-1, 2)).astype(int)
        good = (
            margin[
                np.clip(xy[:, 1], 0, image.shape[0] - 1),
                np.clip(xy[:, 0], 0, image.shape[1] - 1),
            ]
            == 0
        )
        corners, ids = corners[good], ids[good]
        if (
            len(ids) < min_corners
            or np.linalg.matrix_rank(
                object_corners[ids.ravel()] - object_corners[ids.ravel()].mean(axis=0)
            )
            < 2
        ):
            failed_images.append(image_path.name)
            continue
        ids_flat = ids.flatten()
        views.append(
            DetectedView(
                name=image_path.name,
                source_sha256=image_hash(image_path),
                corner_ids=ids_flat.tolist(),
                detection_details=details,
                object_points=object_corners[ids_flat],
                image_points=corners.reshape(-1, 1, 2).astype(np.float32),
                was_rotated=(calibration_image.was_transformed or was_size_normalized),
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
        pixel_policy=pixel_policy,
        board_legacy_pattern=legacy_pattern,
        board_first_marker_id=first_marker_id,
        pattern_size=(squares_x, squares_y),
        square_size=square_size,
        views=views,
        failed_images=failed_images,
        board_type=BOARD_CHARUCO,
        dictionary=dictionary_name,
        marker_proportion=marker_proportion,
    )
