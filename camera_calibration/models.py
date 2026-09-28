"""Central-camera projection in encoded pixels: x right, y down, z forward.

Angular coefficients act on angle in radians. Asymmetric coefficients act on
that angular image plane, not Brown's pinhole plane. No image layout is assumed.
"""

from dataclasses import dataclass

import cv2
import numpy as np

BROWN_MODELS = {"simple", "full", "k1", "plumb_bob"}
ANGULAR_MODELS = {"angular", "angular-asymmetric"}


def radial(theta, coefficients):
    return theta * (
        1 + sum(k * theta ** (2 * i + 2) for i, k in enumerate(coefficients))
    )


def derivative(theta, coefficients):
    return 1 + sum(
        (2 * i + 3) * k * theta ** (2 * i + 2) for i, k in enumerate(coefficients)
    )


def asymmetric_xy(xy, p):
    x, y = xy.T
    p1, p2 = p
    return xy + np.column_stack(
        (
            2 * p1 * x * y + p2 * (3 * x * x + y * y),
            p1 * (x * x + 3 * y * y) + 2 * p2 * x * y,
        )
    )


def asymmetric_jacobian(xy, p):
    x, y = xy.T
    p1, p2 = p
    a = 1 + 2 * p1 * y + 6 * p2 * x
    b = 2 * p1 * x + 2 * p2 * y
    d = 1 + 6 * p1 * y + 2 * p2 * x
    return a, b, d


def tangent_basis(rays):
    z = np.asarray(rays).mean(axis=0)
    if np.linalg.norm(z) < 1e-8:
        raise ValueError("Ray bundle has no stable mean direction")
    z /= np.linalg.norm(z)
    helper = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


@dataclass
class CameraModel:
    matrix: np.ndarray
    coefficients: np.ndarray
    name: str = "full"
    asymmetric: tuple = (0.0, 0.0)
    max_angle_deg: float = 110.0

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=float)
        self.coefficients = np.asarray(self.coefficients, dtype=float).reshape(-1)
        self.asymmetric = np.asarray(self.asymmetric, dtype=float)
        if self.name not in BROWN_MODELS | ANGULAR_MODELS:
            raise ValueError(f"Unsupported projection model: {self.name}")
        if self.matrix.shape != (3, 3) or not np.isfinite(self.matrix).all():
            raise ValueError("Camera matrix must be finite 3x3")
        if self.matrix[0, 0] <= 0 or self.matrix[1, 1] <= 0:
            raise ValueError("Focal scales must be positive")
        if (
            not np.allclose(self.matrix[2], [0, 0, 1])
            or self.matrix[0, 1] != 0
            or self.matrix[1, 0] != 0
        ):
            raise ValueError("Only zero-skew camera matrices are supported")
        if (
            not np.isfinite(self.coefficients).all()
            or not np.isfinite(self.asymmetric).all()
            or self.asymmetric.shape != (2,)
        ):
            raise ValueError("Distortion coefficients must be finite")
        if self.is_angular:
            if not 1 <= len(self.coefficients) <= 4 or not 0 < self.max_angle_deg < 180:
                raise ValueError(
                    "Angular model needs 1–4 radial terms and a half-angle in (0,180)"
                )
            if self.name == "angular" and np.any(self.asymmetric):
                raise ValueError("Asymmetric terms require angular-asymmetric")
        elif np.any(self.asymmetric):
            raise ValueError(
                "Angular-plane asymmetry cannot be attached to a Brown model"
            )
        elif len(self.coefficients) not in {4, 5, 8, 12, 14}:
            raise ValueError("Unsupported Brown coefficient count")

    @property
    def is_angular(self):
        return self.name in ANGULAR_MODELS

    @property
    def focal(self):
        return np.array([self.matrix[0, 0], self.matrix[1, 1]])

    @property
    def center(self):
        return self.matrix[:2, 2]

    def check_domain(self):
        if not self.is_angular:
            return
        theta = np.linspace(0, np.deg2rad(self.max_angle_deg), 4096)
        if np.min(derivative(theta, self.coefficients)) <= 0:
            raise ValueError(
                "Angular radial model is non-monotonic in its declared domain"
            )
        radius = radial(theta[-1], self.coefficients)
        # A conservative sufficient condition for positive definite symmetric
        # Jacobians across the full angular disk (strong monotonicity/injectivity).
        if 6 * np.linalg.norm(self.asymmetric) * radius >= 1:
            raise ValueError(
                "Asymmetric mapping is not certified invertible in its domain"
            )

    def project_rays(self, rays, *, strict=True):
        xyz = np.asarray(rays, dtype=float).reshape(-1, 3)
        if self.is_angular:
            length = np.hypot(xyz[:, 0], xyz[:, 1])
            theta = np.arctan2(length, xyz[:, 2])
            valid = (
                np.isfinite(xyz).all(axis=1)
                & (np.linalg.norm(xyz, axis=1) > 0)
                & (theta <= np.deg2rad(self.max_angle_deg) + 1e-10)
            )
            xy = (
                xyz[:, :2]
                * (radial(theta, self.coefficients) / np.maximum(length, 1e-15))[
                    :, None
                ]
            )
            pixels = asymmetric_xy(xy, self.asymmetric) * self.focal + self.center
        else:
            valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
            pixels = cv2.projectPoints(
                xyz, np.zeros(3), np.zeros(3), self.matrix, self.coefficients
            )[0].reshape(-1, 2)
        valid &= np.isfinite(pixels).all(axis=1)
        if strict and not valid.all():
            raise ValueError("Rays outside projection domain")
        return pixels if strict else (pixels, valid)

    def project(self, objects, pose, *, strict=True):
        pose = np.asarray(pose, dtype=float)
        xyz = (
            np.asarray(objects).reshape(-1, 3) @ cv2.Rodrigues(pose[:3])[0].T + pose[3:]
        )
        return self.project_rays(xyz, strict=strict)

    def unproject_pixels(self, pixels):
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        if not np.isfinite(pixels).all():
            raise ValueError("Pixels must be finite")
        if not self.is_angular:
            xy = cv2.undistortPoints(
                pixels[:, None, :], self.matrix, self.coefficients
            ).reshape(-1, 2)
            rays = np.column_stack((xy, np.ones(len(xy))))
            return rays / np.linalg.norm(rays, axis=1)[:, None]
        self.check_domain()
        target = (pixels - self.center) / self.focal
        xy = target.copy()
        for _ in range(50):
            residual = asymmetric_xy(xy, self.asymmetric) - target
            if np.max(np.abs(residual), initial=0) < 1e-12:
                break
            a, b, d = asymmetric_jacobian(xy, self.asymmetric)
            det = a * d - b * b
            if np.any(det <= 1e-10):
                raise ValueError("Asymmetric inverse crossed a singularity")
            xy -= (
                np.column_stack(
                    (
                        d * residual[:, 0] - b * residual[:, 1],
                        a * residual[:, 1] - b * residual[:, 0],
                    )
                )
                / det[:, None]
            )
        if (
            np.max(np.abs(asymmetric_xy(xy, self.asymmetric) - target), initial=0)
            > 1e-9
        ):
            raise ValueError("Asymmetric inverse failed to converge")
        r = np.linalg.norm(xy, axis=1)
        high = np.full(len(r), np.deg2rad(self.max_angle_deg))
        low = np.zeros(len(r))
        if np.any(r > radial(high, self.coefficients) + 1e-10):
            raise ValueError("Pixels outside angular domain")
        for _ in range(48):
            theta = (low + high) / 2
            greater = radial(theta, self.coefficients) > r
            high = np.where(greater, theta, high)
            low = np.where(greater, low, theta)
        theta = (low + high) / 2
        return np.column_stack(
            (xy * (np.sin(theta) / np.maximum(r, 1e-15))[:, None], np.cos(theta))
        )

    def estimate_board_pose(self, objects, pixels):
        objects = np.asarray(objects, dtype=float).reshape(-1, 3)
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        if (
            len(objects) < 4
            or np.linalg.matrix_rank(objects - objects.mean(axis=0)) < 2
        ):
            raise ValueError("Need at least four noncollinear target points")
        if not self.is_angular:
            ok, rv, tv = cv2.solvePnP(objects, pixels, self.matrix, self.coefficients)
            if not ok:
                raise ValueError("Board pose estimation failed")
            return np.r_[rv.ravel(), tv.ravel()]
        from scipy.optimize import least_squares

        rays = self.unproject_pixels(pixels)
        basis = tangent_basis(rays)
        virtual = rays @ basis
        if np.any(virtual[:, 2] <= 0.05):
            raise ValueError(
                "Board spans too wide an angle for tangent pose initialization"
            )
        ok, rv, tv = cv2.solvePnP(
            objects, virtual[:, :2] / virtual[:, 2:], np.eye(3), None
        )
        if not ok:
            raise ValueError("Board pose initialization failed")
        seed = np.r_[
            cv2.Rodrigues(basis @ cv2.Rodrigues(rv)[0])[0].ravel(), (basis @ tv).ravel()
        ]
        fit = least_squares(
            lambda p: (self.project(objects, p, strict=False)[0] - pixels).ravel(),
            seed,
            loss="soft_l1",
            f_scale=2,
            x_scale="jac",
            max_nfev=200,
        )
        if not fit.success:
            raise ValueError("Board pose refinement did not converge")
        self.project(objects, fit.x)
        return fit.x


def model_for(calibration):
    return CameraModel(
        calibration.camera_matrix,
        calibration.distortion_coefficients,
        calibration.distortion_model,
        calibration.asymmetric_coefficients,
        calibration.max_angle_deg,
    )
