"""Fitzgibbon one-parameter division distortion model (λ).

Alongside Brown–Conrady (OpenCV plumb_bob), this estimates a single radial
parameter λ in the division model (Fitzgibbon 2001):

    x_d = x_u / (1 + λ * r_u²)
    y_d = y_u / (1 + λ * r_u²)

where (x_u, y_u) are ideal pinhole coordinates in the normalized camera plane
(z=1), r_u² = x_u² + y_u², and (x_d, y_d) are the distorted normalized
coordinates before applying K.

Convention: λ > 0 typically corresponds to barrel-like behaviour under this
mapping (|x_d| < |x_u| for r_u > 0 when λ > 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detection import DetectionSet


@dataclass
class FitzgibbonEstimate:
    """Fitted division-model parameter and its reprojection RMS."""

    lambda_: float
    rms_reprojection_error: float


def _division_distort_normalized(xy_u: np.ndarray, lambda_: float) -> np.ndarray:
    r2 = np.sum(xy_u * xy_u, axis=1)
    denom = 1.0 + lambda_ * r2
    # Guard against pathological λ that zeros the denominator.
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    return xy_u / denom[:, None]


def _project_with_lambda(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    lambda_: float,
) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(rvec)
    points_cam = (rotation @ object_points.T).T + tvec.reshape(1, 3)
    # Normalized pinhole coordinates.
    xy_u = points_cam[:, :2] / points_cam[:, 2:3]
    xy_d = _division_distort_normalized(xy_u, lambda_)
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    pixels = np.empty_like(xy_d)
    pixels[:, 0] = fx * xy_d[:, 0] + cx
    pixels[:, 1] = fy * xy_d[:, 1] + cy
    return pixels


def _poses_zero_distortion(
    detections: DetectionSet,
    camera_matrix: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-view poses with K fixed and no Brown–Conrady distortion."""
    zero_dist = np.zeros((5, 1), dtype=np.float64)
    poses: list[tuple[np.ndarray, np.ndarray]] = []
    for view in detections.views:
        ok, rvec, tvec = cv2.solvePnP(
            view.object_points,
            view.image_points,
            camera_matrix,
            zero_dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError(f"solvePnP failed for {view.name} while fitting λ")
        poses.append((rvec, tvec))
    return poses


def _rms_for_lambda(
    detections: DetectionSet,
    camera_matrix: np.ndarray,
    poses: list[tuple[np.ndarray, np.ndarray]],
    lambda_: float,
) -> float:
    sq_sum = 0.0
    count = 0
    for view, (rvec, tvec) in zip(detections.views, poses):
        projected = _project_with_lambda(
            view.object_points, rvec, tvec, camera_matrix, lambda_
        )
        observed = view.image_points.reshape(-1, 2)
        delta = projected - observed
        sq_sum += float(np.sum(delta * delta))
        count += len(observed)
    return math_sqrt(sq_sum / max(count, 1))


def math_sqrt(value: float) -> float:
    return float(np.sqrt(value))


def _brent_minimize(
    objective,
    lo: float,
    hi: float,
    tol: float = 1e-10,
    max_iter: int = 80,
) -> float:
    """1D Brent-style minimization without SciPy (λ is scalar)."""
    # Golden-section search — robust enough for this smooth 1D residual.
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    resphi = 2.0 - phi
    a, b = lo, hi
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = objective(x1)
    f2 = objective(x2)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = a + resphi * (b - a)
            f1 = objective(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = b - resphi * (b - a)
            f2 = objective(x2)
    return float(x1 if f1 < f2 else x2)


def estimate_fitzgibbon_lambda(
    detections: DetectionSet,
    camera_matrix: np.ndarray,
    search_lo: float = -1.0,
    search_hi: float = 1.0,
) -> FitzgibbonEstimate:
    """
    Fit Fitzgibbon λ with K fixed (from Brown–Conrady / OpenCV calib).

    Poses are estimated with zero Brown distortion so λ carries the radial
    residual. Search is 1D over a wide bracket of λ.
    """
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    poses = _poses_zero_distortion(detections, camera_matrix)

    def objective(lambda_: float) -> float:
        return _rms_for_lambda(detections, camera_matrix, poses, float(lambda_))

    # Coarse grid to place a bracket, then refine.
    grid = np.linspace(search_lo, search_hi, 41)
    grid_rms = [objective(float(value)) for value in grid]
    best_idx = int(np.argmin(grid_rms))
    lo = float(grid[max(0, best_idx - 1)])
    hi = float(grid[min(len(grid) - 1, best_idx + 1)])
    if lo == hi:
        lo, hi = search_lo, search_hi

    lambda_hat = _brent_minimize(objective, lo, hi)
    rms = objective(lambda_hat)
    return FitzgibbonEstimate(lambda_=float(lambda_hat), rms_reprojection_error=float(rms))
