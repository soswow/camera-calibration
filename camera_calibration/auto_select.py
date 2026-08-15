"""Select a robust, diverse subset of calibration views after a first fit."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detection import DetectionSet, fit_intrinsics
from .result import CalibrationResult


@dataclass
class ViewScore:
    name: str
    mean_error_px: float
    max_error_px: float
    center_x: float
    center_y: float
    span: float
    distance: float
    tilt_deg: float


@dataclass
class AutoSelectResult:
    kept_names: list[str]
    rejected_names: list[str]
    rejection_reasons: dict[str, str]
    initial_rms: float
    error_threshold_px: float


def _tilt_deg(rvec: np.ndarray) -> float:
    rotation, _ = cv2.Rodrigues(rvec)
    normal = rotation[:, 2]
    cos_tilt = float(np.clip(abs(float(np.dot(normal, [0.0, 0.0, 1.0]))), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_tilt)))


def score_views(
    detections: DetectionSet,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[ViewScore]:
    """Per-view reprojection stats and pose descriptors from a fitted model."""
    width, height = detections.image_size
    scores: list[ViewScore] = []

    for view in detections.views:
        ok, rvec, tvec = cv2.solvePnP(
            view.object_points,
            view.image_points,
            camera_matrix,
            dist_coeffs,
        )
        if not ok:
            scores.append(
                ViewScore(
                    name=view.name,
                    mean_error_px=1e9,
                    max_error_px=1e9,
                    center_x=0.5,
                    center_y=0.5,
                    span=0.0,
                    distance=0.0,
                    tilt_deg=0.0,
                )
            )
            continue

        projected, _ = cv2.projectPoints(
            view.object_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        residuals = view.image_points.reshape(-1, 2) - projected.reshape(-1, 2)
        norms = np.linalg.norm(residuals, axis=1)
        points = view.image_points.reshape(-1, 2)
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        center = (min_xy + max_xy) / 2.0
        span = (max_xy - min_xy) / np.array([width, height], dtype=np.float64)

        scores.append(
            ViewScore(
                name=view.name,
                mean_error_px=float(norms.mean()),
                max_error_px=float(norms.max()),
                center_x=float(center[0] / width),
                center_y=float(center[1] / height),
                span=float(max(span[0], span[1])),
                distance=float(np.linalg.norm(tvec)),
                tilt_deg=_tilt_deg(rvec),
            )
        )

    return scores


def _coverage_key(score: ViewScore, dist_edges: tuple[float, float]) -> tuple[int, int, int, int]:
    """Coarse pose/location bucket used to preserve diversity."""
    spatial_x = min(2, int(score.center_x * 3))
    spatial_y = min(2, int(score.center_y * 3))
    if score.tilt_deg < 20:
        tilt_bin = 0
    elif score.tilt_deg < 40:
        tilt_bin = 1
    else:
        tilt_bin = 2

    low, high = dist_edges
    if score.distance < low:
        dist_bin = 0
    elif score.distance < high:
        dist_bin = 1
    else:
        dist_bin = 2
    return spatial_x, spatial_y, tilt_bin, dist_bin


def select_views(
    scores: list[ViewScore],
    initial_rms: float,
    *,
    error_factor: float = 1.5,
    error_floor_px: float = 2.0,
    max_keep: int | None = None,
    min_keep: int = 10,
) -> AutoSelectResult:
    """
    Reject high-residual outliers, then keep a diverse inlier subset.

    Diversity: prefer covering distinct (frame location × tilt × distance) buckets,
    taking the lowest-error view in each bucket first, then fill by rising error.
    """
    if not scores:
        raise RuntimeError("No scored views available for auto-select")

    threshold = max(error_floor_px, error_factor * initial_rms)
    reasons: dict[str, str] = {}
    inliers: list[ViewScore] = []
    rejected: list[str] = []

    for score in scores:
        # Extreme close+tilt with high residual — often dominates RMS.
        hard_close_tilt = score.tilt_deg >= 40 and score.span >= 0.55
        if score.mean_error_px > threshold:
            rejected.append(score.name)
            reasons[score.name] = (
                f"mean error {score.mean_error_px:.2f}px > threshold {threshold:.2f}px"
            )
            continue
        if hard_close_tilt and score.mean_error_px > max(initial_rms, np.median([s.mean_error_px for s in scores])):
            rejected.append(score.name)
            reasons[score.name] = (
                f"close+steep tilt (tilt={score.tilt_deg:.0f}°, span={score.span:.2f}) "
                f"with elevated error {score.mean_error_px:.2f}px"
            )
            continue
        inliers.append(score)

    # If we were too aggressive, fall back to residual-only rejection.
    if len(inliers) < min(min_keep, max(3, len(scores) // 3)):
        inliers = [score for score in scores if score.mean_error_px <= threshold]
        rejected = [score.name for score in scores if score.mean_error_px > threshold]
        reasons = {
            score.name: f"mean error {score.mean_error_px:.2f}px > threshold {threshold:.2f}px"
            for score in scores
            if score.mean_error_px > threshold
        }

    if len(inliers) < 3:
        # Keep the best residuals so calibration can still run.
        ordered = sorted(scores, key=lambda item: item.mean_error_px)
        inliers = ordered[: max(3, min(min_keep, len(ordered)))]
        kept = {score.name for score in inliers}
        rejected = [score.name for score in scores if score.name not in kept]
        reasons = {
            name: "kept only lowest-error views (too few inliers after filtering)"
            for name in rejected
        }

    distances = [score.distance for score in inliers]
    if len(distances) >= 3:
        dist_edges = (float(np.percentile(distances, 33)), float(np.percentile(distances, 66)))
    else:
        dist_edges = (float(min(distances)), float(max(distances)))

    # Diversity pass: one best view per coverage bucket, then fill by error.
    inliers_sorted = sorted(inliers, key=lambda item: item.mean_error_px)
    bucket_best: dict[tuple[int, int, int, int], ViewScore] = {}
    for score in inliers_sorted:
        key = _coverage_key(score, dist_edges)
        if key not in bucket_best:
            bucket_best[key] = score

    selected: list[ViewScore] = list(bucket_best.values())
    selected_names = {score.name for score in selected}
    for score in inliers_sorted:
        if score.name in selected_names:
            continue
        selected.append(score)
        selected_names.add(score.name)

    if max_keep is not None and len(selected) > max_keep:
        # Keep bucket representatives first (already at front), then lowest error.
        selected = selected[:max_keep]
        selected_names = {score.name for score in selected}
        for score in inliers:
            if score.name not in selected_names:
                rejected.append(score.name)
                reasons[score.name] = f"dropped for diversity budget (max_keep={max_keep})"

    # Anything in inliers but not selected was capped by max_keep; already handled.
    kept_names = [score.name for score in selected]
    # Preserve stable ordering by rising error for reproducibility in reports.
    kept_names = [
        score.name
        for score in sorted(selected, key=lambda item: item.mean_error_px)
    ]

    return AutoSelectResult(
        kept_names=kept_names,
        rejected_names=sorted(set(rejected)),
        rejection_reasons=reasons,
        initial_rms=initial_rms,
        error_threshold_px=threshold,
    )


def filter_detections(detections: DetectionSet, kept_names: list[str]) -> DetectionSet:
    """Return a DetectionSet containing only the named views."""
    keep = set(kept_names)
    views = [view for view in detections.views if view.name in keep]
    if len(views) < 3:
        raise RuntimeError(f"Auto-select kept only {len(views)} views; need at least 3")
    return DetectionSet(
        image_size=detections.image_size,
        pattern_size=detections.pattern_size,
        square_size=detections.square_size,
        views=views,
        failed_images=list(detections.failed_images),
        board_type=detections.board_type,
        dictionary=detections.dictionary,
        marker_proportion=detections.marker_proportion,
    )


def auto_select_and_refit(
    detections: DetectionSet,
    distortion_model: str = "simple",
    *,
    error_factor: float = 1.5,
    error_floor_px: float = 2.0,
    max_keep: int | None = None,
    min_keep: int = 10,
) -> tuple[CalibrationResult, AutoSelectResult]:
    """
    Fit on all detections, select a robust diverse subset, refit.

    Returns (CalibrationResult, AutoSelectResult).
    """
    initial = fit_intrinsics(detections, distortion_model=distortion_model)
    camera_matrix = np.asarray(initial.camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(initial.distortion_coefficients, dtype=np.float64)

    scores = score_views(detections, camera_matrix, dist_coeffs)
    selection = select_views(
        scores,
        initial.rms_reprojection_error,
        error_factor=error_factor,
        error_floor_px=error_floor_px,
        max_keep=max_keep,
        min_keep=min_keep,
    )

    filtered = filter_detections(detections, selection.kept_names)
    refined = fit_intrinsics(filtered, distortion_model=distortion_model)

    # Annotate selection metadata on the result object.
    refined.failed_images = list(
        dict.fromkeys([*detections.failed_images, *selection.rejected_names])
    )
    refined.used_images = selection.kept_names
    refined.auto_select_rejected = selection.rejected_names
    refined.auto_select_threshold_px = selection.error_threshold_px
    refined.initial_rms_reprojection_error = selection.initial_rms
    refined.rotated_images = [
        view.name for view in filtered.views if view.was_rotated
    ]

    return refined, selection
