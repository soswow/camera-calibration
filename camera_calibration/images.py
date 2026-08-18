"""Image listing and EXIF-orientation helpers shared by tools."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXIF_ORIENTATION_TAG = 274
DETECTION_ROTATIONS = (0, 90, 180, 270)


@dataclass(frozen=True)
class CalibrationImage:
    """Image pixels normalized into the calibration coordinate frame."""

    image: np.ndarray
    exif_orientation: int
    was_transformed: bool


def imread_sensor(path: Path | str) -> np.ndarray | None:
    """
    Read an image without applying EXIF orientation.

    Calibration estimates intrinsics in the pixel coordinate frame passed to
    OpenCV. For phone photos, EXIF display rotation can differ between shots
    even though the sensor/lens coordinate frame is fixed, so calibration must
    use the encoded pixels rather than display-oriented pixels.
    """
    flags = cv2.IMREAD_COLOR
    if hasattr(cv2, "IMREAD_IGNORE_ORIENTATION"):
        flags |= cv2.IMREAD_IGNORE_ORIENTATION
    return cv2.imread(str(path), flags)


def read_exif_orientation(path: Path | str) -> int:
    """Return the EXIF Orientation value, defaulting to 1 when absent/invalid."""
    try:
        with Image.open(path) as image:
            orientation = image.getexif().get(EXIF_ORIENTATION_TAG, 1)
    except Exception:
        return 1

    try:
        orientation = int(orientation)
    except (TypeError, ValueError):
        return 1

    if 1 <= orientation <= 8:
        return orientation
    return 1


def apply_exif_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
    """
    Apply an EXIF Orientation transform to an OpenCV image.

    EXIF orientation values describe the transform from stored raster to display
    orientation. The inverse transform is used for calibration normalization.
    """
    if orientation == 1:
        return image
    if orientation == 2:
        return cv2.flip(image, 1)
    if orientation == 3:
        return cv2.rotate(image, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(image, 0)
    if orientation == 5:
        return cv2.transpose(image)
    if orientation == 6:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.transpose(image), -1)
    if orientation == 8:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def inverse_exif_orientation(orientation: int) -> int:
    """Return the EXIF orientation value that reverses `orientation`."""
    return {6: 8, 8: 6}.get(orientation, orientation)


def normalize_for_calibration(
    image: np.ndarray,
    exif_orientation: int,
) -> tuple[np.ndarray, bool]:
    """
    Return pixels in the calibration frame.

    The calibration frame is the inverse of EXIF display orientation: it undoes
    orientation metadata intended for viewing so that photos taken with different
    phone rotations can share one camera pixel coordinate frame.
    """
    inverse_orientation = inverse_exif_orientation(exif_orientation)
    normalized = apply_exif_orientation(image, inverse_orientation)
    return normalized, inverse_orientation != 1


def read_calibration_image(path: Path | str) -> CalibrationImage | None:
    """Read image pixels and normalize them into the calibration frame."""
    image = imread_sensor(path)
    if image is None:
        return None

    exif_orientation = read_exif_orientation(path)
    normalized, was_transformed = normalize_for_calibration(image, exif_orientation)
    return CalibrationImage(
        image=normalized,
        exif_orientation=exif_orientation,
        was_transformed=was_transformed,
    )


def list_images(folder: Path) -> list[Path]:
    """Return sorted image paths under folder (non-recursive)."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def choose_canonical_image_size(images: list[Path]) -> tuple[int, int]:
    """
    Pick the (width, height) used for calibration.

    Images are normalized with inverse EXIF display orientation before sizing.
    A consistent calibration needs all accepted views in one fixed pixel
    coordinate frame.
    """
    counts: Counter[tuple[int, int]] = Counter()
    for image_path in images:
        calibration_image = read_calibration_image(image_path)
        if calibration_image is None:
            continue
        image = calibration_image.image
        height, width = image.shape[:2]
        counts[(width, height)] += 1

    if not counts:
        raise FileNotFoundError("Could not read any images to determine size")

    # Prefer the most frequent size; break ties toward portrait (height >= width).
    return max(counts.items(), key=lambda item: (item[1], item[0][1] >= item[0][0]))[0]


def require_image_size(
    image: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray | None:
    """Return image only when it already matches target (width, height)."""
    height, width = image.shape[:2]
    if (width, height) == target_size:
        return image
    return None


def normalize_to_calibration_size(
    image: np.ndarray,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, bool] | None:
    """
    Return image in target size, allowing only exact portrait/landscape transpose.

    EXIF normalization is preferred. This fallback handles exports where display
    rotation has already been baked into pixels and EXIF Orientation is reset.
    """
    height, width = image.shape[:2]
    target_width, target_height = target_size
    if (width, height) == (target_width, target_height):
        return image, False
    if (width, height) == (target_height, target_width):
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), True
    return None


def rotate_for_detection(image: np.ndarray, rotation_degrees: int) -> np.ndarray:
    """Rotate a temporary detection image clockwise by 0/90/180/270 degrees."""
    if rotation_degrees == 0:
        return image
    if rotation_degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation_degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation_degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported detection rotation: {rotation_degrees}")


def unrotate_detection_points(
    points: np.ndarray,
    original_size: tuple[int, int],
    rotation_degrees: int,
) -> np.ndarray:
    """Map points from a temporary rotated detection image to the original image."""
    if rotation_degrees == 0:
        return points

    width, height = original_size
    restored = points.copy()
    xy = restored.reshape(-1, 2)
    x = xy[:, 0].copy()
    y = xy[:, 1].copy()

    if rotation_degrees == 90:
        xy[:, 0] = y
        xy[:, 1] = (height - 1) - x
    elif rotation_degrees == 180:
        xy[:, 0] = (width - 1) - x
        xy[:, 1] = (height - 1) - y
    elif rotation_degrees == 270:
        xy[:, 0] = (width - 1) - y
        xy[:, 1] = x
    else:
        raise ValueError(f"Unsupported detection rotation: {rotation_degrees}")

    return restored
