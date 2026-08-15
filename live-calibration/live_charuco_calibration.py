#!/usr/bin/env python3
"""Live ChArUco calibration helper with preview + coverage heatmap."""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
import threading
from queue import Empty, Queue

# ---- Configuration ----
DEVICE_INDEX = 0
WIDTH = 640
HEIGHT = 480
FPS = 30
FORMAT = "bgr"  # "bgr" or "raw"
QUEUE_SIZE = 2
TIMEOUT_MS = 1000
# Flip the incoming image horizontally (mirror view).
FLIP_HORIZONTAL = True
# Beep on each accepted frame.
BEEP_ON_ACCEPT = True
# Minimum seconds between beeps to avoid audio spam.
BEEP_MIN_INTERVAL = 0.2
# System beep sound on macOS (set to None to disable).
MAC_BEEP_SOUND = "/System/Library/Sounds/Funk.aiff"

# Exposure settings
AUTO_EXPOSURE = False
GAIN = 5
EXPOSURE = 160

# ChArUco board settings (match your board)
SQUARES_X = 7
SQUARES_Y = 10
SQUARE_SIZE_MM = 79.14
MARKER_PROPORTION = 0.7
DICTIONARY = "DICT_4X4_50"

# Frame acceptance
# Minimum number of ArUco markers required before attempting ChArUco interpolation.
# Higher values reduce false positives but make detection more strict.
MIN_MARKERS = 2
# Minimum number of ChArUco corners required to consider a frame valid for capture.
# Raise this to demand richer corner coverage per frame.
MIN_CHARUCO = 6
# Laplacian-variance sharpness threshold on the board ROI; lower allows blurrier frames.
# Set to 0 to disable sharpness filtering entirely.
MIN_SHARPNESS = 15
# Minimum fraction of the image area covered by the board's bounding box.
# Prevents accepting tiny, far-away boards.
MIN_AREA_FRAC = 0.02
# Maximum fraction of the image area covered by the board's bounding box.
# Prevents accepting boards that are too close or clipped.
MAX_AREA_FRAC = 0.90

# Calibration
# Minimum total number of accepted frames before calibration is attempted.
# More frames generally improves stability but takes longer to collect.
CALIB_MIN_FRAMES = 20
# Run calibration after every N new accepted frames.
# Increase to reduce calibration frequency and UI load.
CALIB_EVERY_N_ACCEPT = 5
# Per-view reprojection error threshold for pruning outlier frames (in pixels).
# Set to 0 to disable pruning and keep all accepted frames.
PRUNE_THRESHOLD = 1.5  # pixels; set 0 to disable pruning
# Minimum number of ChArUco corners required to include a frame in calibration.
# Frames with fewer corners are ignored for calibration to avoid degenerate views.
CALIB_MIN_CHARUCO = 6
# Cap total number of stored observations to keep calibration fast.
# When the cap is reached, higher-scoring frames may replace lower-scoring ones.
MAX_FRAMES = 200
# Maximum number of observations allowed per heatmap cell (diversity cap).
# Set to 0 to disable per-cell capping.
PER_CELL_MAX = 5
# Diversity scoring
# Prefer frames that cover low-coverage areas; higher values bias selection more strongly.
COVERAGE_GAIN_WEIGHT = 0.2
# Weight for the number of detected ChArUco corners in the score.
CORNER_COUNT_WEIGHT = 0.40
# Pose diversity
# Enable pose-aware diversity once calibration parameters are available.
POSE_DIVERSITY = True
# Pose bin size in degrees (pitch/yaw/roll).
POSE_BIN_DEG = 10
# Maximum number of observations allowed per pose bin.
# Set to 0 to disable pose bin capping.
POSE_PER_BIN_MAX = 5
# Weight for pose novelty in the score.
POSE_GAIN_WEIGHT = 0.8

# Diversity capping geometry
# Scale applied to the board polygon for diversity caps (1.0 = full polygon).
# Smaller values make caps less restrictive when the board is very close.
CAP_POLY_SCALE = 0.5

# Hold-out validation
# Fraction of calibration-eligible frames to reserve for validation.
HOLDOUT_FRACTION = 0.15
# Minimum number of hold-out views (if possible).
HOLDOUT_MIN = 20
# Maximum number of hold-out views (0 = no cap).
HOLDOUT_MAX = 50
# Deterministic seed for hold-out splitting.
HOLDOUT_SEED = 1338
# Show hold-out error in the status line.
SHOW_HOLDOUT = True

# Heatmap
# Heatmap grid width (in cells), used to visualize board coverage across the frame.
# Increase for finer spatial resolution at the cost of more noise.
HEATMAP_W = WIDTH
# Heatmap grid height (in cells), used to visualize board coverage across the frame.
# Increase for finer spatial resolution at the cost of more noise.
HEATMAP_H = HEIGHT
# Heatmap display
# Overlay the heatmap on top of the preview window (semi-transparent).
HEATMAP_OVERLAY = True
# Alpha for the heatmap overlay (0 = invisible, 1 = full heatmap).
HEATMAP_OVERLAY_ALPHA = 0.3
# Show a separate heatmap-only window.
SHOW_HEATMAP_WINDOW = False

# Residual scatter plot (dx vs dy)
# Show residual scatter plot in a separate window.
SHOW_RESIDUAL_SCATTER = True
# Max absolute residual (pixels) shown on the scatter axes.
RESIDUAL_SCATTER_RANGE_PX = 2.0
# Scatter image size in pixels.
RESIDUAL_SCATTER_SIZE = 320
# Minimum number of residual samples to draw scatter.
RESIDUAL_SCATTER_MIN_SAMPLES = 30
# Maximum number of residual samples to plot (0 = no cap).
RESIDUAL_SCATTER_MAX_SAMPLES = 2000

# Saving
# Write calibration parameters to a JSON file whenever a new calibration result arrives.
# Uses atomic replace to avoid partial files if the app exits mid-write.
AUTO_SAVE_CALIB = True
AUTO_SAVE_PATH = "output/live_calibration_latest.json"
# Dump a full session snapshot (including observations) when quitting with 'q' or Esc.
DUMP_ON_EXIT = True
DUMP_ON_EXIT_PATH = "output/live_calibration_dump.json"
# Include all accepted observations in the exit dump for offline recalibration.
# Set to False to save only parameters and metadata on exit.
DUMP_INCLUDE_OBSERVATIONS = True

# Loading
# Load a previous session to continue collecting from existing observations.
# Tries the detailed dump first, then the latest-parameters file.
LOAD_ON_STARTUP = True
LOAD_PATHS = (DUMP_ON_EXIT_PATH, AUTO_SAVE_PATH)
# Refilter loaded observations using current acceptance thresholds.
# Only checks MIN_CHARUCO and area bounds (sharpness/markers can't be recovered).
REFILTER_ON_LOAD = True
# Run a calibration pass on loaded data to refresh reprojection error and enable pruning.
CALIBRATE_ON_LOAD = True

# Undistorted preview
# Show a third window with live grayscale frames undistorted using the latest calibration.
SHOW_UNDISTORTED = True
# Alpha passed to getOptimalNewCameraMatrix (0 = crop to valid pixels, 1 = keep full FOV).
UNDISTORT_ALPHA = 0.5
# -----------------------


def _get_dictionary(aruco, name: str):
    if not hasattr(aruco, name):
        available = [k for k in dir(aruco) if k.startswith("DICT_")]
        raise SystemExit(
            f"Unknown dictionary '{name}'. Available examples: {', '.join(sorted(available)[:8])}"
        )
    return aruco.getPredefinedDictionary(getattr(aruco, name))


def _create_board(aruco, squares_x: int, squares_y: int, square_size: float, marker_size: float, dictionary):
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


def _create_detector(aruco, dictionary):
    if not hasattr(aruco, "ArucoDetector"):
        return None
    if hasattr(aruco, "DetectorParameters"):
        params = aruco.DetectorParameters()
        return aruco.ArucoDetector(dictionary, params)
    return aruco.ArucoDetector(dictionary)


def _detect_markers(aruco, gray, dictionary, detector=None):
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = aruco.detectMarkers(gray, dictionary)
    return corners, ids


def _sharpness(gray, bbox) -> float:
    import cv2

    if bbox:
        x0, y0, x1, y1 = bbox
        roi = gray[y0 : y1 + 1, x0 : x1 + 1]
    else:
        roi = gray
    if roi.size == 0:
        return 0.0
    lap = cv2.Laplacian(roi, cv2.CV_64F)
    return float(lap.var())


def _compute_per_view_errors(board, all_corners, all_ids, rvecs, tvecs, camera_matrix, dist_coeffs):
    import cv2
    import numpy as np

    corners = _get_board_corners(board)
    per_view = []
    for idx, (charuco_corners, charuco_ids) in enumerate(zip(all_corners, all_ids)):
        ids = charuco_ids.flatten()
        obj_pts = corners[ids].astype(np.float32)
        img_pts = charuco_corners.reshape(-1, 2).astype(np.float32)
        if len(obj_pts) < 4:
            per_view.append(float("inf"))
            continue
        proj, _ = cv2.projectPoints(
            obj_pts, rvecs[idx], tvecs[idx], camera_matrix, dist_coeffs
        )
        proj = proj.reshape(-1, 2)
        diff = img_pts - proj
        err = (diff * diff).sum(axis=1).mean() ** 0.5
        per_view.append(float(err))
    return per_view


def _get_board_corners(board):
    if hasattr(board, "getChessboardCorners"):
        return board.getChessboardCorners()
    return board.chessboardCorners


def _homography_ok(cv2, board_corners, charuco_corners, charuco_ids, np):
    if charuco_corners is None or charuco_ids is None:
        return False
    if len(charuco_corners) < CALIB_MIN_CHARUCO or len(charuco_ids) < CALIB_MIN_CHARUCO:
        return False
    ids = charuco_ids.flatten()
    if len(ids) != len(charuco_corners):
        return False
    obj_pts = board_corners[ids].astype(np.float32)
    img_pts = charuco_corners.reshape(-1, 2).astype(np.float32)
    if len(obj_pts) < 4:
        return False
    homography, _ = cv2.findHomography(obj_pts[:, :2], img_pts, 0)
    return homography is not None and homography.shape == (3, 3)


def _atomic_write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def _maybe_beep(last_beep_time: float) -> float:
    now = time.monotonic()
    if now - last_beep_time < BEEP_MIN_INTERVAL:
        return last_beep_time
    if sys.platform == "darwin" and MAC_BEEP_SOUND:
        try:
            subprocess.Popen(
                ["afplay", MAC_BEEP_SOUND],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            print("\a", end="", flush=True)
    else:
        print("\a", end="", flush=True)
    return now


def _split_holdout(n: int):
    if HOLDOUT_FRACTION <= 0:
        return list(range(n)), []
    min_required = max(HOLDOUT_MIN * 2, 8)
    if n < min_required:
        return list(range(n)), []
    holdout_size = int(round(n * HOLDOUT_FRACTION))
    holdout_size = max(holdout_size, HOLDOUT_MIN)
    if HOLDOUT_MAX > 0:
        holdout_size = min(holdout_size, HOLDOUT_MAX)
    holdout_size = min(holdout_size, n - 1)
    if n - holdout_size < CALIB_MIN_FRAMES:
        holdout_size = max(0, n - CALIB_MIN_FRAMES)
    if holdout_size <= 0:
        return list(range(n)), []
    rng = random.Random(HOLDOUT_SEED)
    indices = list(range(n))
    rng.shuffle(indices)
    holdout_set = set(indices[:holdout_size])
    train = [i for i in range(n) if i not in holdout_set]
    holdout = sorted(holdout_set)
    return train, holdout


def _serialize_observations(all_corners, all_ids):
    observations = []
    for corners, ids in zip(all_corners, all_ids):
        if corners is None or ids is None:
            continue
        corners_xy = corners.reshape(-1, 2).tolist()
        ids_list = [int(i) for i in ids.flatten().tolist()]
        observations.append({"ids": ids_list, "corners": corners_xy})
    return observations


def _build_calibration_payload(
    image_size,
    frames_used,
    reproj_error,
    camera_matrix,
    dist_coeffs,
    per_view_errors,
    all_corners,
    all_ids,
    include_observations: bool,
    holdout_error=None,
    holdout_used=None,
):
    payload = {
        "timestamp_unix": time.time(),
        "image_size": list(image_size) if image_size is not None else None,
        "board": {
            "squares_x": SQUARES_X,
            "squares_y": SQUARES_Y,
            "square_size_mm": SQUARE_SIZE_MM,
            "marker_size_mm": SQUARE_SIZE_MM * MARKER_PROPORTION,
            "dictionary": DICTIONARY,
        },
        "frames_used": frames_used,
        "reprojection_error": reproj_error,
        "camera_matrix": camera_matrix.tolist() if camera_matrix is not None else None,
        "dist_coeffs": dist_coeffs.tolist() if dist_coeffs is not None else None,
        "distortion_coefficients": (
            dist_coeffs.ravel().tolist() if dist_coeffs is not None else None
        ),
        "rms_reprojection_error": reproj_error,
        "board_type": "charuco",
        "pattern_size": [SQUARES_X, SQUARES_Y],
        "square_size": SQUARE_SIZE_MM,
        "dictionary": DICTIONARY,
        "marker_proportion": MARKER_PROPORTION,
        "per_view_errors": per_view_errors,
        "acceptance": {
            "min_markers": MIN_MARKERS,
            "min_charuco": MIN_CHARUCO,
            "min_sharpness": MIN_SHARPNESS,
            "min_area_frac": MIN_AREA_FRAC,
            "max_area_frac": MAX_AREA_FRAC,
        },
        "calibration": {
            "calib_min_frames": CALIB_MIN_FRAMES,
            "calib_every_n_accept": CALIB_EVERY_N_ACCEPT,
            "prune_threshold": PRUNE_THRESHOLD,
            "calib_min_charuco": CALIB_MIN_CHARUCO,
        },
        "validation": {
            "holdout_fraction": HOLDOUT_FRACTION,
            "holdout_min": HOLDOUT_MIN,
            "holdout_max": HOLDOUT_MAX,
            "holdout_error": holdout_error,
            "holdout_views": holdout_used,
        },
    }
    if include_observations:
        payload["observations"] = _serialize_observations(all_corners, all_ids)
    return payload


def _cell_polygon(corners, image_size, np):
    import cv2

    if image_size is None:
        return None
    width, height = image_size
    if width <= 0 or height <= 0:
        return None
    pts = corners.reshape(-1, 2).astype(np.float32)
    if not np.isfinite(pts).all():
        return None
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    hx = box[:, 0] / float(width) * HEATMAP_W
    hy = box[:, 1] / float(height) * HEATMAP_H
    poly = np.stack([hx, hy], axis=1)
    poly[:, 0] = np.clip(poly[:, 0], 0, HEATMAP_W - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, HEATMAP_H - 1)
    poly = poly.astype(np.int32)
    if len(poly) < 3:
        return None
    return poly


def _cap_polygon(poly):
    import numpy as np

    if poly is None:
        return None
    scale = float(CAP_POLY_SCALE)
    if scale >= 0.999:
        return poly
    if scale <= 0.0:
        center = np.round(poly.mean(axis=0)).astype(np.int32)
        return center.reshape(1, 2)
    center = poly.mean(axis=0)
    scaled = center + (poly - center) * scale
    scaled = np.round(scaled).astype(np.int32)
    scaled[:, 0] = np.clip(scaled[:, 0], 0, HEATMAP_W - 1)
    scaled[:, 1] = np.clip(scaled[:, 1], 0, HEATMAP_H - 1)
    return scaled


def _apply_cell_polygon(grid, poly, delta):
    import cv2
    import numpy as np

    if poly is None:
        return
    mask = np.zeros_like(grid, dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 1)
    grid[mask == 1] += delta


def _rebuild_cap_counts(all_cells):
    import numpy as np

    counts = np.zeros((HEATMAP_H, HEATMAP_W), dtype=np.int32)
    for poly in all_cells:
        cap_poly = _cap_polygon(poly)
        _apply_cell_polygon(counts, cap_poly, 1)
    return counts


def _rebuild_heatmap(all_cells):
    import numpy as np

    hm = np.zeros((HEATMAP_H, HEATMAP_W), dtype=np.float32)
    for poly in all_cells:
        _apply_cell_polygon(hm, poly, 1.0)
    return hm


def _draw_residual_scatter(samples):
    import cv2
    import numpy as np

    size = int(RESIDUAL_SCATTER_SIZE)
    size = max(64, size)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    span = max(1e-6, float(RESIDUAL_SCATTER_RANGE_PX))
    scale = (size / 2 - 1) / span
    # Axes
    cv2.line(img, (0, center), (size - 1, center), (60, 60, 60), 1, cv2.LINE_AA)
    cv2.line(img, (center, 0), (center, size - 1), (60, 60, 60), 1, cv2.LINE_AA)
    # Points
    for dx, dy in samples:
        if abs(dx) > span or abs(dy) > span:
            continue
        x = int(round(center + dx * scale))
        y = int(round(center - dy * scale))
        if 0 <= x < size and 0 <= y < size:
            img[y, x] = (0, 255, 0)
    return img


def _compute_residual_samples(aruco, cv2, board, all_corners, all_ids, camera_matrix, dist_coeffs):
    import numpy as np

    if camera_matrix is None or dist_coeffs is None:
        return []
    board_corners = _get_board_corners(board)
    samples = []
    for corners, ids in zip(all_corners, all_ids):
        if corners is None or ids is None:
            continue
        if len(corners) < 6 or len(ids) < 6:
            continue
        if len(corners) != len(ids):
            continue
        try:
            retval, rvec, tvec = aruco.estimatePoseCharucoBoard(
                corners, ids, board, camera_matrix, dist_coeffs, None, None
            )
        except Exception:
            continue
        if not retval:
            continue
        ids_flat = ids.flatten()
        obj_pts = board_corners[ids_flat].astype(np.float32)
        img_pts = corners.reshape(-1, 2).astype(np.float32)
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
        proj = proj.reshape(-1, 2)
        diff = img_pts - proj
        for dx, dy in diff:
            samples.append((float(dx), float(dy)))
    if RESIDUAL_SCATTER_MAX_SAMPLES > 0 and len(samples) > RESIDUAL_SCATTER_MAX_SAMPLES:
        rng = random.Random(HOLDOUT_SEED)
        samples = rng.sample(samples, RESIDUAL_SCATTER_MAX_SAMPLES)
    return samples


def _coverage_gain(counts, poly):
    import cv2
    import numpy as np

    if poly is None:
        return 0.0
    mask = np.zeros_like(counts, dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 1)
    region = counts[mask == 1]
    if region.size == 0:
        return 0.0
    return float((1.0 / (1.0 + region)).sum())


def _pose_gain(pose_counts, pose_bin):
    if pose_bin is None:
        return 0.0
    return 1.0 / (1.0 + pose_counts.get(pose_bin, 0))


def _score_frame(counts, poly, num_corners, pose_counts=None, pose_bin=None):
    score = CORNER_COUNT_WEIGHT * float(num_corners)
    if COVERAGE_GAIN_WEIGHT > 0:
        score += COVERAGE_GAIN_WEIGHT * _coverage_gain(counts, poly)
    if POSE_GAIN_WEIGHT > 0 and pose_counts is not None:
        score += POSE_GAIN_WEIGHT * _pose_gain(pose_counts, pose_bin)
    return score


def _can_add_cells(counts, poly, per_cell_max):
    import cv2
    import numpy as np

    if per_cell_max <= 0:
        return True
    if poly is None:
        return False
    mask = np.zeros_like(counts, dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 1)
    region = counts[mask == 1]
    if region.size == 0:
        return True
    return int(region.max()) < per_cell_max


def _can_add_pose(pose_counts, pose_bin, per_bin_max):
    if per_bin_max <= 0 or pose_bin is None:
        return True
    return pose_counts.get(pose_bin, 0) < per_bin_max


def _rvec_to_euler_deg(cv2, rvec):
    rot, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0])
    singular = sy < 1.0e-6
    if not singular:
        pitch = math.atan2(rot[2, 1], rot[2, 2])
        yaw = math.atan2(-rot[2, 0], sy)
        roll = math.atan2(rot[1, 0], rot[0, 0])
    else:
        pitch = math.atan2(-rot[1, 2], rot[1, 1])
        yaw = math.atan2(-rot[2, 0], sy)
        roll = 0.0
    return (
        math.degrees(pitch),
        math.degrees(yaw),
        math.degrees(roll),
    )


def _pose_bin_from_rvec(cv2, rvec):
    if POSE_BIN_DEG <= 0:
        return None
    pitch, yaw, roll = _rvec_to_euler_deg(cv2, rvec)
    return (
        int(math.floor(pitch / POSE_BIN_DEG)),
        int(math.floor(yaw / POSE_BIN_DEG)),
        int(math.floor(roll / POSE_BIN_DEG)),
    )


def _estimate_pose_bin(aruco, cv2, corners, ids, board, camera_matrix, dist_coeffs):
    if (
        not POSE_DIVERSITY
        or camera_matrix is None
        or dist_coeffs is None
        or corners is None
        or ids is None
    ):
        return None
    if not hasattr(aruco, "estimatePoseCharucoBoard"):
        return None
    if len(corners) < 6 or len(ids) < 6:
        return None
    try:
        retval, rvec, _ = aruco.estimatePoseCharucoBoard(
            corners, ids, board, camera_matrix, dist_coeffs, None, None
        )
    except Exception:
        return None
    if not retval:
        return None
    return _pose_bin_from_rvec(cv2, rvec)


def _compute_pose_bins(all_corners, all_ids, aruco, cv2, board, camera_matrix, dist_coeffs):
    pose_bins = []
    for corners, ids in zip(all_corners, all_ids):
        pose_bins.append(_estimate_pose_bin(aruco, cv2, corners, ids, board, camera_matrix, dist_coeffs))
    return pose_bins


def _filter_loaded_observations(all_corners, all_ids, image_size, np, board_corners=None, cv2=None):
    filtered_corners = []
    filtered_ids = []
    if image_size is not None:
        width, height = image_size
    else:
        width = None
        height = None
    for corners, ids in zip(all_corners, all_ids):
        if corners is None or ids is None:
            continue
        required_charuco = max(MIN_CHARUCO, CALIB_MIN_CHARUCO)
        if len(corners) < required_charuco:
            continue
        if len(ids) != len(corners):
            continue
        if width is not None and height is not None:
            pts = corners.reshape(-1, 2)
            if not np.isfinite(pts).all():
                continue
            span_x = float(pts[:, 0].max() - pts[:, 0].min())
            span_y = float(pts[:, 1].max() - pts[:, 1].min())
            area_frac = (span_x * span_y) / float(width * height) if width * height > 0 else 0.0
            if area_frac < MIN_AREA_FRAC or area_frac > MAX_AREA_FRAC:
                continue
        if board_corners is not None and cv2 is not None:
            if not _homography_ok(cv2, board_corners, corners, ids, np):
                continue
        filtered_corners.append(corners)
        filtered_ids.append(ids)
    return filtered_corners, filtered_ids


def _calib_indices(all_corners, all_ids):
    indices = []
    for idx, (corners, ids) in enumerate(zip(all_corners, all_ids)):
        if corners is None or ids is None:
            continue
        if len(corners) >= CALIB_MIN_CHARUCO and len(ids) >= CALIB_MIN_CHARUCO:
            indices.append(idx)
    return indices


def _select_best_diverse(all_corners, all_ids, image_size, pose_bins, np):
    import cv2

    max_frames = len(all_corners) if MAX_FRAMES <= 0 else MAX_FRAMES
    base_scores = []
    poly_list = []
    cap_poly_list = []
    for corners in all_corners:
        poly = _cell_polygon(corners, image_size, np)
        poly_list.append(poly)
        if poly is None:
            cap_poly_list.append(None)
            base_scores.append(float("-inf"))
            continue
        cap_poly = _cap_polygon(poly)
        cap_poly_list.append(cap_poly)
        area_cells = float(cv2.contourArea(poly)) if len(poly) >= 3 else 0.0
        base_scores.append(CORNER_COUNT_WEIGHT * float(len(corners)) + COVERAGE_GAIN_WEIGHT * area_cells)
    order = sorted(range(len(all_corners)), key=lambda idx: base_scores[idx], reverse=True)
    selected_corners = []
    selected_ids = []
    selected_scores = []
    selected_cells = []
    selected_pose_bins = []
    counts = np.zeros((HEATMAP_H, HEATMAP_W), dtype=np.int32)
    pose_counts = {}
    for idx in order:
        corners = all_corners[idx]
        ids = all_ids[idx]
        poly = poly_list[idx]
        cap_poly = cap_poly_list[idx]
        pose_bin = pose_bins[idx] if pose_bins is not None else None
        if poly is None:
            continue
        if not _can_add_cells(counts, cap_poly, PER_CELL_MAX):
            continue
        if not _can_add_pose(pose_counts, pose_bin, POSE_PER_BIN_MAX):
            continue
        score = _score_frame(counts, cap_poly, len(corners), pose_counts, pose_bin)
        selected_corners.append(corners)
        selected_ids.append(ids)
        selected_scores.append(score)
        selected_cells.append(poly)
        selected_pose_bins.append(pose_bin)
        _apply_cell_polygon(counts, cap_poly, 1)
        if pose_bin is not None:
            pose_counts[pose_bin] = pose_counts.get(pose_bin, 0) + 1
        if len(selected_corners) >= max_frames:
            break
    return selected_corners, selected_ids, selected_scores, selected_cells, counts, selected_pose_bins, pose_counts


def _load_session_payload(paths, np):
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        observations = data.get("observations") or []
        loaded_corners = []
        loaded_ids = []
        for entry in observations:
            corners = entry.get("corners")
            ids = entry.get("ids")
            if not corners or not ids:
                continue
            corners_arr = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
            ids_arr = np.array(ids, dtype=np.int32).reshape(-1, 1)
            if len(corners_arr) != len(ids_arr):
                continue
            loaded_corners.append(corners_arr)
            loaded_ids.append(ids_arr)

        camera_matrix = data.get("camera_matrix")
        dist_coeffs = data.get("dist_coeffs")
        payload = {
            "path": path,
            "corners": loaded_corners,
            "ids": loaded_ids,
            "image_size": tuple(data["image_size"]) if data.get("image_size") else None,
            "reprojection_error": data.get("reprojection_error"),
            "camera_matrix": np.array(camera_matrix, dtype=np.float64) if camera_matrix else None,
            "dist_coeffs": np.array(dist_coeffs, dtype=np.float64) if dist_coeffs else None,
            "frames_used": data.get("frames_used"),
            "per_view_errors": data.get("per_view_errors"),
        }
        return payload
    return None


def _calibration_worker(aruco, board, requests: Queue, results: Queue):
    import cv2
    import numpy as np

    while True:
        payload = requests.get()
        if payload is None:
            break
        all_corners, all_ids, image_size, indices = payload
        result = None
        try:
            board_corners = _get_board_corners(board)
            filtered_corners = []
            filtered_ids = []
            filtered_indices = []
            for corners, ids, idx in zip(all_corners, all_ids, indices):
                if not _homography_ok(cv2, board_corners, corners, ids, np):
                    continue
                filtered_corners.append(corners)
                filtered_ids.append(ids)
                filtered_indices.append(idx)
            if len(filtered_corners) < CALIB_MIN_FRAMES:
                result = {
                    "error": "calibration_failed",
                    "message": f"Not enough valid views for calibration: {len(filtered_corners)}",
                }
                raise cv2.error("Calibration aborted")
            train_idx, holdout_idx = _split_holdout(len(filtered_corners))
            train_corners = [filtered_corners[i] for i in train_idx]
            train_ids = [filtered_ids[i] for i in train_idx]
            train_indices = [filtered_indices[i] for i in train_idx]
            holdout_corners = [filtered_corners[i] for i in holdout_idx]
            holdout_ids = [filtered_ids[i] for i in holdout_idx]
            if hasattr(aruco, "calibrateCameraCharucoExtended"):
                (
                    ret,
                    camera_matrix,
                    dist_coeffs,
                    rvecs,
                    tvecs,
                    _,
                    _,
                    per_view_errors,
                ) = aruco.calibrateCameraCharucoExtended(
                    train_corners, train_ids, board, image_size, None, None
                )
            else:
                ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
                    train_corners, train_ids, board, image_size, None, None
                )
                per_view_errors = _compute_per_view_errors(
                    board, train_corners, train_ids, rvecs, tvecs, camera_matrix, dist_coeffs
                )
            if per_view_errors is not None:
                per_view_errors = [float(np.asarray(e).reshape(-1)[0]) for e in per_view_errors]
            holdout_error = None
            holdout_used = 0
            if holdout_corners and holdout_ids:
                errors = []
                for corners, ids in zip(holdout_corners, holdout_ids):
                    try:
                        retval, rvec, tvec = aruco.estimatePoseCharucoBoard(
                            corners, ids, board, camera_matrix, dist_coeffs, None, None
                        )
                    except Exception:
                        continue
                    if not retval:
                        continue
                    ids_flat = ids.flatten()
                    obj_pts = board_corners[ids_flat].astype(np.float32)
                    img_pts = corners.reshape(-1, 2).astype(np.float32)
                    proj, _ = cv2.projectPoints(
                        obj_pts, rvec, tvec, camera_matrix, dist_coeffs
                    )
                    proj = proj.reshape(-1, 2)
                    diff = img_pts - proj
                    err = (diff * diff).sum(axis=1).mean() ** 0.5
                    errors.append(float(err))
                if errors:
                    holdout_error = sum(errors) / len(errors)
                    holdout_used = len(errors)
            residual_samples = []
            for corners, ids, rvec, tvec in zip(train_corners, train_ids, rvecs, tvecs):
                ids_flat = ids.flatten()
                obj_pts = board_corners[ids_flat].astype(np.float32)
                img_pts = corners.reshape(-1, 2).astype(np.float32)
                proj, _ = cv2.projectPoints(
                    obj_pts, rvec, tvec, camera_matrix, dist_coeffs
                )
                proj = proj.reshape(-1, 2)
                diff = img_pts - proj
                for dx, dy in diff:
                    residual_samples.append((float(dx), float(dy)))
            if RESIDUAL_SCATTER_MAX_SAMPLES > 0 and len(residual_samples) > RESIDUAL_SCATTER_MAX_SAMPLES:
                rng = random.Random(HOLDOUT_SEED)
                residual_samples = rng.sample(residual_samples, RESIDUAL_SCATTER_MAX_SAMPLES)
            result = (
                float(ret),
                camera_matrix,
                dist_coeffs,
                per_view_errors,
                train_indices,
                image_size,
                holdout_error,
                holdout_used,
                residual_samples,
            )
        except cv2.error as exc:
            if result is None:
                result = {"error": "calibration_failed", "message": str(exc)}

        try:
            results.get_nowait()
        except Empty:
            pass
        if result is not None:
            results.put(result)
        requests.task_done()


def _build_undistort_maps(cv2, camera_matrix, dist_coeffs, image_size):
    width, height = image_size
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), UNDISTORT_ALPHA, (width, height)
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
        (width, height),
        cv2.CV_16SC2,
    )
    return map1, map2


def main() -> int:
    try:
        import ps3eye  # type: ignore
    except Exception as exc:
        print("Failed to import ps3eye. Ensure the module is on PYTHONPATH.")
        print(f"Error: {exc}")
        return 1

    try:
        import cv2
        import numpy as np
        from cv2 import aruco
    except Exception as exc:
        print("OpenCV is required.")
        print(f"Error: {exc}")
        return 1

    devices = ps3eye.list_devices()
    if not devices:
        print("No PS3 Eye devices found.")
        return 1

    dictionary = _get_dictionary(aruco, DICTIONARY)
    marker_size = SQUARE_SIZE_MM * MARKER_PROPORTION
    board = _create_board(aruco, SQUARES_X, SQUARES_Y, SQUARE_SIZE_MM, marker_size, dictionary)
    detector = _create_detector(aruco, dictionary)

    all_corners = []
    all_ids = []
    all_scores = []
    all_cells = []
    all_pose_bins = []
    accepted_positions = []
    accepted_since_calib = 0
    last_reproj = None
    last_reproj_label = "Reproj"
    last_camera_matrix = None
    last_dist_coeffs = None
    last_per_view_errors = None
    last_calib_indices = None
    last_image_size = None
    last_frames_used = None
    last_holdout_error = None
    last_holdout_used = None
    last_residual_samples = []
    coverage_counts = np.zeros((HEATMAP_H, HEATMAP_W), dtype=np.int32)
    pose_counts = {}
    calib_pending = False
    last_beep_time = 0.0
    undistort_map1 = None
    undistort_map2 = None
    undistort_size = None
    if LOAD_ON_STARTUP:
        session = _load_session_payload(LOAD_PATHS, np)
        if session is not None:
            all_corners = session["corners"]
            all_ids = session["ids"]
            last_reproj = session["reprojection_error"]
            if last_reproj is not None:
                last_reproj_label = "Reproj (loaded)"
            last_camera_matrix = session["camera_matrix"]
            last_dist_coeffs = session["dist_coeffs"]
            last_per_view_errors = session["per_view_errors"]
            last_calib_indices = None
            last_image_size = session["image_size"]
            last_frames_used = session["frames_used"] if session["frames_used"] is not None else len(all_corners)
            if last_image_size is None:
                last_image_size = (WIDTH, HEIGHT)
            if REFILTER_ON_LOAD:
                before = len(all_corners)
                board_corners = _get_board_corners(board)
                all_corners, all_ids = _filter_loaded_observations(
                    all_corners, all_ids, last_image_size, np, board_corners, cv2
                )
                if len(all_corners) != before:
                    last_reproj = None
                    last_per_view_errors = None
                    last_calib_indices = None
                    last_reproj_label = "Reproj"
                    print(f"Refiltered observations: {before} -> {len(all_corners)}")
            before_select = len(all_corners)
            pose_bins = None
            if POSE_DIVERSITY and last_camera_matrix is not None and last_dist_coeffs is not None:
                pose_bins = _compute_pose_bins(
                    all_corners, all_ids, aruco, cv2, board, last_camera_matrix, last_dist_coeffs
                )
            (
                all_corners,
                all_ids,
                all_scores,
                all_cells,
                coverage_counts,
                all_pose_bins,
                pose_counts,
            ) = _select_best_diverse(all_corners, all_ids, last_image_size, pose_bins, np)
            if len(all_corners) != before_select:
                last_reproj = None
                last_per_view_errors = None
                last_calib_indices = None
                last_reproj_label = "Reproj"
                print(f"Selected {len(all_corners)} of {before_select} observations after capping")
            last_frames_used = len(all_corners)
            print(f"Loaded {len(all_corners)} observations from {session['path']}")
    calib_requests: Queue = Queue(maxsize=1)
    calib_results: Queue = Queue(maxsize=1)
    calib_thread = threading.Thread(
        target=_calibration_worker,
        args=(aruco, board, calib_requests, calib_results),
        daemon=True,
    )
    calib_thread.start()

    heatmap = _rebuild_heatmap(all_cells)
    if all_corners:
        if last_image_size is None:
            last_image_size = (WIDTH, HEIGHT)
        accepted_positions = [
            corners.reshape(-1, 2).mean(axis=0) for corners in all_corners
        ]
    if (
        SHOW_UNDISTORTED
        and last_camera_matrix is not None
        and last_dist_coeffs is not None
        and last_image_size is not None
    ):
        undistort_map1, undistort_map2 = _build_undistort_maps(
            cv2, last_camera_matrix, last_dist_coeffs, last_image_size
        )
        undistort_size = last_image_size
    if (
        CALIBRATE_ON_LOAD
        and not calib_requests.full()
    ):
        if last_image_size is None:
            last_image_size = (WIDTH, HEIGHT)
        calib_indices = _calib_indices(all_corners, all_ids)
        if len(calib_indices) >= CALIB_MIN_FRAMES:
            calib_corners = [all_corners[i] for i in calib_indices]
            calib_ids = [all_ids[i] for i in calib_indices]
            calib_requests.put((calib_corners, calib_ids, last_image_size, calib_indices))
            calib_pending = True

    with ps3eye.Device(DEVICE_INDEX) as dev:
        dev.configure(width=WIDTH, height=HEIGHT, fps=FPS, format=FORMAT)
        if AUTO_EXPOSURE:
            dev.set_auto(True)
        else:
            dev.set_auto(False)
            dev.set_manual(GAIN, EXPOSURE)
        dev.start_stream(queue_size=QUEUE_SIZE)

        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
        if SHOW_HEATMAP_WINDOW:
            cv2.namedWindow("coverage", cv2.WINDOW_NORMAL)
        if SHOW_UNDISTORTED:
            cv2.namedWindow("undistorted", cv2.WINDOW_NORMAL)
        if SHOW_RESIDUAL_SCATTER:
            cv2.namedWindow("residuals", cv2.WINDOW_NORMAL)

        while True:
            frame = dev.read(timeout_ms=TIMEOUT_MS)
            if frame is None:
                continue

            img = frame.to_numpy(copy=False)
            if img.ndim == 2:
                gray = img
                color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                color = img
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            last_image_size = (gray.shape[1], gray.shape[0])

            marker_corners, marker_ids = _detect_markers(aruco, gray, dictionary, detector)
            num_markers = 0 if marker_ids is None else int(len(marker_ids))
            found_charuco = False
            charuco_corners = None
            charuco_ids = None
            sharp = 0.0
            area_frac = 0.0

            required_charuco = max(MIN_CHARUCO, CALIB_MIN_CHARUCO)
            if marker_ids is not None and len(marker_ids) >= MIN_MARKERS:
                try:
                    _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                        marker_corners, marker_ids, gray, board
                    )
                except cv2.error:
                    charuco_corners = None
                    charuco_ids = None

                if charuco_corners is not None and len(charuco_corners) >= required_charuco:
                    found_charuco = True
                    rect = cv2.minAreaRect(charuco_corners)
                    box = cv2.boxPoints(rect).astype(int)
                    cv2.polylines(color, [box], True, (255, 0, 0), 2)
                    x, y, w, h = cv2.boundingRect(charuco_corners.astype(int))
                    sharp = _sharpness(gray, (x, y, x + w - 1, y + h - 1))
                    area_frac = (w * h) / float(gray.shape[0] * gray.shape[1])

            accepted = False
            if found_charuco:
                if (
                    sharp >= MIN_SHARPNESS
                    and MIN_AREA_FRAC <= area_frac <= MAX_AREA_FRAC
                ):
                    accepted = True

            poly = None
            cap_poly = None
            pose_bin = None
            score = 0.0
            if accepted:
                poly = _cell_polygon(charuco_corners, (gray.shape[1], gray.shape[0]), np)
                if poly is not None:
                    cap_poly = _cap_polygon(poly)
                    pose_bin = _estimate_pose_bin(
                        aruco, cv2, charuco_corners, charuco_ids, board, last_camera_matrix, last_dist_coeffs
                    )
                    score = _score_frame(
                        coverage_counts, cap_poly, len(charuco_corners), pose_counts, pose_bin
                    )
                if cap_poly is None or not _can_add_cells(coverage_counts, cap_poly, PER_CELL_MAX):
                    accepted = False
                elif not _can_add_pose(pose_counts, pose_bin, POSE_PER_BIN_MAX):
                    accepted = False
                else:
                    if not _homography_ok(cv2, _get_board_corners(board), charuco_corners, charuco_ids, np):
                        accepted = False

            if accepted and MAX_FRAMES > 0 and len(all_corners) >= MAX_FRAMES:
                worst_idx = min(range(len(all_scores)), key=lambda i: all_scores[i])
                if score <= all_scores[worst_idx]:
                    accepted = False
                else:
                    old_cap = _cap_polygon(all_cells[worst_idx])
                    _apply_cell_polygon(coverage_counts, old_cap, -1)
                    _apply_cell_polygon(heatmap, all_cells[worst_idx], -1.0)
                    if all_pose_bins:
                        old_pose = all_pose_bins[worst_idx]
                        if old_pose is not None:
                            pose_counts[old_pose] = max(0, pose_counts.get(old_pose, 1) - 1)
                    if not _can_add_cells(coverage_counts, cap_poly, PER_CELL_MAX):
                        _apply_cell_polygon(coverage_counts, old_cap, 1)
                        _apply_cell_polygon(heatmap, all_cells[worst_idx], 1.0)
                        if all_pose_bins:
                            old_pose = all_pose_bins[worst_idx]
                            if old_pose is not None:
                                pose_counts[old_pose] = pose_counts.get(old_pose, 0) + 1
                        accepted = False
                    elif not _can_add_pose(pose_counts, pose_bin, POSE_PER_BIN_MAX):
                        _apply_cell_polygon(coverage_counts, old_cap, 1)
                        _apply_cell_polygon(heatmap, all_cells[worst_idx], 1.0)
                        if all_pose_bins:
                            old_pose = all_pose_bins[worst_idx]
                            if old_pose is not None:
                                pose_counts[old_pose] = pose_counts.get(old_pose, 0) + 1
                        accepted = False
                    else:
                        all_corners.pop(worst_idx)
                        all_ids.pop(worst_idx)
                        all_scores.pop(worst_idx)
                        all_cells.pop(worst_idx)
                        if all_pose_bins:
                            all_pose_bins.pop(worst_idx)
                        if accepted_positions:
                            accepted_positions.pop(worst_idx)

            if accepted:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                all_scores.append(score)
                all_cells.append(poly)
                all_pose_bins.append(pose_bin)
                _apply_cell_polygon(coverage_counts, cap_poly, 1)
                _apply_cell_polygon(heatmap, poly, 1.0)
                if pose_bin is not None:
                    pose_counts[pose_bin] = pose_counts.get(pose_bin, 0) + 1
                accepted_since_calib += 1
                center = charuco_corners.reshape(-1, 2).mean(axis=0)
                accepted_positions.append(center)
                if BEEP_ON_ACCEPT:
                    last_beep_time = _maybe_beep(last_beep_time)

            # Pull async calibration results.
            try:
                result = calib_results.get_nowait()
            except Empty:
                result = None
            if result is not None:
                if isinstance(result, dict) and result.get("error") == "calibration_failed":
                    calib_pending = False
                    last_reproj = None
                    last_reproj_label = "Reproj"
                    last_holdout_error = None
                    last_holdout_used = None
                    last_residual_samples = []
                    message = result.get("message", "unknown error")
                    print(f"Calibration failed (cv2.error): {message}")
                else:
                    (
                        last_reproj,
                        last_camera_matrix,
                        last_dist_coeffs,
                        last_per_view_errors,
                        last_calib_indices,
                        last_image_size,
                        last_holdout_error,
                        last_holdout_used,
                        last_residual_samples,
                    ) = result
                    calib_pending = False
                    last_reproj_label = "Reproj"
                    last_frames_used = len(last_calib_indices)
                if (
                    PRUNE_THRESHOLD > 0
                    and last_per_view_errors is not None
                    and last_calib_indices is not None
                    and len(last_calib_indices) == len(last_per_view_errors)
                ):
                    keep_mask = [True] * len(all_corners)
                    for idx, err in zip(last_calib_indices, last_per_view_errors):
                        if err > PRUNE_THRESHOLD and idx < len(keep_mask):
                            keep_mask[idx] = False
                    keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
                    all_corners = [all_corners[i] for i in keep_indices]
                    all_ids = [all_ids[i] for i in keep_indices]
                    all_scores = [all_scores[i] for i in keep_indices]
                    all_cells = [all_cells[i] for i in keep_indices]
                    all_pose_bins = [all_pose_bins[i] for i in keep_indices]
                    accepted_positions = [accepted_positions[i] for i in keep_indices]
                    coverage_counts = _rebuild_cap_counts(all_cells)
                    pose_counts = {}
                    for pose_bin in all_pose_bins:
                        if pose_bin is not None:
                            pose_counts[pose_bin] = pose_counts.get(pose_bin, 0) + 1
                    heatmap = _rebuild_heatmap(all_cells)
                if (
                    last_per_view_errors is not None
                    and last_calib_indices is not None
                    and MAX_FRAMES > 0
                    and len(all_corners) > MAX_FRAMES
                ):
                    errors_by_index = [1.0e6] * len(all_corners)
                    for idx, err in zip(last_calib_indices, last_per_view_errors):
                        if idx < len(errors_by_index):
                            errors_by_index[idx] = err
                    ranked = []
                    for idx in range(len(all_corners)):
                        ranked.append((errors_by_index[idx], -all_scores[idx], idx))
                    ranked.sort()
                    keep_indices = sorted(idx for _, _, idx in ranked[:MAX_FRAMES])
                    all_corners = [all_corners[i] for i in keep_indices]
                    all_ids = [all_ids[i] for i in keep_indices]
                    all_scores = [all_scores[i] for i in keep_indices]
                    all_cells = [all_cells[i] for i in keep_indices]
                    all_pose_bins = [all_pose_bins[i] for i in keep_indices]
                    accepted_positions = [accepted_positions[i] for i in keep_indices]
                    coverage_counts = _rebuild_cap_counts(all_cells)
                    pose_counts = {}
                    for pose_bin in all_pose_bins:
                        if pose_bin is not None:
                            pose_counts[pose_bin] = pose_counts.get(pose_bin, 0) + 1
                    heatmap = _rebuild_heatmap(all_cells)
                if AUTO_SAVE_CALIB and last_camera_matrix is not None and last_dist_coeffs is not None:
                    payload = _build_calibration_payload(
                        last_image_size,
                        last_frames_used,
                        last_reproj,
                        last_camera_matrix,
                        last_dist_coeffs,
                        last_per_view_errors,
                        all_corners,
                        all_ids,
                        include_observations=False,
                        holdout_error=last_holdout_error,
                        holdout_used=last_holdout_used,
                    )
                    _atomic_write_json(AUTO_SAVE_PATH, payload)
                if POSE_DIVERSITY and last_camera_matrix is not None and last_dist_coeffs is not None:
                    pose_bins = _compute_pose_bins(
                        all_corners, all_ids, aruco, cv2, board, last_camera_matrix, last_dist_coeffs
                    )
                    before_pose = len(all_corners)
                    (
                        all_corners,
                        all_ids,
                        all_scores,
                        all_cells,
                        coverage_counts,
                        all_pose_bins,
                        pose_counts,
                    ) = _select_best_diverse(all_corners, all_ids, last_image_size, pose_bins, np)
                    if len(all_corners) != before_pose:
                        last_per_view_errors = None
                        last_calib_indices = None
                        print(f"Pose-diverse selection: {before_pose} -> {len(all_corners)}")
                    accepted_positions = [
                        corners.reshape(-1, 2).mean(axis=0) for corners in all_corners
                    ]
                    heatmap = _rebuild_heatmap(all_cells)
                    last_frames_used = len(all_corners)
                if SHOW_RESIDUAL_SCATTER and last_camera_matrix is not None and last_dist_coeffs is not None:
                    residual_samples = _compute_residual_samples(
                        aruco,
                        cv2,
                        board,
                        all_corners,
                        all_ids,
                        last_camera_matrix,
                        last_dist_coeffs,
                    )
                    if residual_samples:
                        last_residual_samples = residual_samples
                else:
                    last_residual_samples = []
                if (
                    SHOW_UNDISTORTED
                    and last_camera_matrix is not None
                    and last_dist_coeffs is not None
                    and last_image_size is not None
                ):
                    undistort_map1, undistort_map2 = _build_undistort_maps(
                        cv2, last_camera_matrix, last_dist_coeffs, last_image_size
                    )
                    undistort_size = last_image_size

            # Calibrate periodically (async to avoid UI stalls).
            if (
                accepted_since_calib >= CALIB_EVERY_N_ACCEPT
            ):
                if not calib_requests.full():
                    calib_indices = _calib_indices(all_corners, all_ids)
                    if len(calib_indices) >= CALIB_MIN_FRAMES:
                        accepted_since_calib = 0
                        calib_corners = [all_corners[i] for i in calib_indices]
                        calib_ids = [all_ids[i] for i in calib_indices]
                        calib_requests.put(
                            (calib_corners, calib_ids, (gray.shape[1], gray.shape[0]), calib_indices)
                        )
                        calib_pending = True

            # Overlay text (drawn after optional flip so text reads normally).
            if last_reproj is not None:
                status_text = f"Accepted: {len(all_corners)}  {last_reproj_label}: {last_reproj:.3f}"
                if SHOW_HOLDOUT and last_holdout_error is not None and last_holdout_used:
                    status_text += f"  Holdout: {last_holdout_error:.3f} ({last_holdout_used})"
            elif calib_pending:
                status_text = f"Accepted: {len(all_corners)}  Reproj: calibrating..."
            else:
                status_text = f"Accepted: {len(all_corners)}  Reproj: N/A"

            # Heatmap window.
            hm = heatmap.copy()
            if hm.max() > 0:
                hm = hm / hm.max()
            hm_img = (hm * 255).astype(np.uint8)
            hm_color = cv2.applyColorMap(hm_img, cv2.COLORMAP_JET)
            hm_color = cv2.resize(hm_color, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

            if HEATMAP_OVERLAY:
                overlay = cv2.addWeighted(
                    color, 1.0 - HEATMAP_OVERLAY_ALPHA, hm_color, HEATMAP_OVERLAY_ALPHA, 0.0
                )
                display_preview = overlay
            else:
                display_preview = color
            if FLIP_HORIZONTAL:
                display_preview = cv2.flip(display_preview, 1)
            cv2.putText(
                display_preview,
                status_text,
                (12, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_preview,
                status_text,
                (12, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if accepted:
                cv2.putText(
                    display_preview,
                    "ACCEPTED",
                    (12, display_preview.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 200, 0),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("preview", display_preview)
            if SHOW_HEATMAP_WINDOW:
                display_heatmap = hm_color
                if FLIP_HORIZONTAL:
                    display_heatmap = cv2.flip(display_heatmap, 1)
                cv2.imshow("coverage", display_heatmap)
            if SHOW_UNDISTORTED:
                if (
                    undistort_map1 is not None
                    and undistort_map2 is not None
                    and undistort_size == (gray.shape[1], gray.shape[0])
                ):
                    undistorted = cv2.remap(
                        gray, undistort_map1, undistort_map2, interpolation=cv2.INTER_LINEAR
                    )
                else:
                    undistorted = gray.copy()
                if FLIP_HORIZONTAL:
                    undistorted = cv2.flip(undistorted, 1)
                if undistort_map1 is None or undistort_map2 is None:
                    cv2.putText(
                        undistorted,
                        "No calibration",
                        (12, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        255,
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow("undistorted", undistorted)
            if SHOW_RESIDUAL_SCATTER:
                if len(last_residual_samples) >= RESIDUAL_SCATTER_MIN_SAMPLES:
                    scatter_img = _draw_residual_scatter(last_residual_samples)
                else:
                    scatter_img = _draw_residual_scatter([])
                    cv2.putText(
                        scatter_img,
                        "No residuals",
                        (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (200, 200, 200),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.imshow("residuals", scatter_img)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        dev.stop_stream()
        if DUMP_ON_EXIT:
            payload = _build_calibration_payload(
                last_image_size,
                last_frames_used,
                last_reproj,
                last_camera_matrix,
                last_dist_coeffs,
                last_per_view_errors,
                all_corners,
                all_ids,
                include_observations=DUMP_INCLUDE_OBSERVATIONS,
                holdout_error=last_holdout_error,
                holdout_used=last_holdout_used,
            )
            _atomic_write_json(DUMP_ON_EXIT_PATH, payload)
        try:
            calib_requests.put_nowait(None)
        except Exception:
            pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
