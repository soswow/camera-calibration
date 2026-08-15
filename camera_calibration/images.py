"""Image listing and orientation helpers shared by calibrate / undistort / diagnose."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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

    Phone sets often mix portrait and landscape of the same sensor
    (e.g. 2160x3840 and 3840x2160). We pick the more common orientation
    and later rotate the others to match.
    """
    counts: Counter[tuple[int, int]] = Counter()
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        counts[(width, height)] += 1

    if not counts:
        raise FileNotFoundError("Could not read any images to determine size")

    # Prefer the most frequent size; break ties toward portrait (height >= width).
    return max(counts.items(), key=lambda item: (item[1], item[0][1] >= item[0][0]))[0]


def orient_image_to_size(
    image: np.ndarray,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, bool] | None:
    """
    Return image in target (width, height), rotating 90° if needed.

    Returns None when the frame is a true different resolution (not just rotated).
    """
    height, width = image.shape[:2]
    target_width, target_height = target_size
    if (width, height) == (target_width, target_height):
        return image, False
    if (width, height) == (target_height, target_width):
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), True
    return None
