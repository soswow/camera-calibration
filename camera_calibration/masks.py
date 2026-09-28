"""Generic per-image exclusion polygons in normalized image coordinates."""

import json
from pathlib import Path

import cv2
import numpy as np


def load_masks(path):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Masks must map image names to polygon lists")
    return data


def mask_for(shape, polygons):
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 3
            or not np.isfinite(points).all()
            or np.any((points < 0) | (points > 1))
        ):
            raise ValueError(
                "Mask polygon requires at least three finite normalized XY vertices"
            )
        cv2.fillPoly(mask, [np.rint(points * [w - 1, h - 1]).astype(np.int32)], 255)
    return mask
