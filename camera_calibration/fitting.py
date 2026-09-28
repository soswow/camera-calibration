"""Robust single-camera angular bundle adjustment on ordinary image observations."""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .models import CameraModel, derivative


def fit_angular(
    views,
    image_size,
    *,
    terms=2,
    asymmetric=False,
    seed_fov=195.0,
    max_angle=110.0,
    max_nfev=500,
):
    if not 1 <= terms <= 4 or len(views) < 8:
        raise ValueError(
            "Angular fitting requires 1–4 radial terms and at least eight distinct views"
        )
    if not 0 < seed_fov < 2 * max_angle < 360:
        raise ValueError("Require 0 < seed FOV < twice maximum half-angle < 360")
    width, height = image_size
    size = min(width, height)
    n = 4 + terms + (2 if asymmetric else 0)
    f = size / np.deg2rad(seed_fov)
    initial = np.r_[f, f, (width - 1) / 2, (height - 1) / 2, np.zeros(n - 4)]
    name = "angular-asymmetric" if asymmetric else "angular"

    def model(x):
        return CameraModel(
            [[x[0], 0, x[2]], [0, x[1], x[3]], [0, 0, 1]],
            x[4 : 4 + terms],
            name,
            x[-2:] if asymmetric else (0.0, 0.0),
            max_angle,
        )

    if asymmetric:
        seed, seed_poses, _ = fit_angular(
            views,
            image_size,
            terms=terms,
            seed_fov=seed_fov,
            max_angle=max_angle,
            max_nfev=max_nfev,
        )
        initial[: 4 + terms] = seed
        poses = seed_poses
    else:
        seed = model(initial)
        poses = [
            seed.estimate_board_pose(v.object_points, v.image_points) for v in views
        ]
    grid = np.linspace(0, np.deg2rad(max_angle), 80)

    def residual(x):
        m = model(x[:n])
        data = [
            (
                m.project(v.object_points, x[n + 6 * i : n + 6 * i + 6], strict=False)[
                    0
                ]
                - v.image_points.reshape(-1, 2)
            ).ravel()
            for i, v in enumerate(views)
        ]
        data.append(1000 * np.minimum(derivative(grid, x[4 : 4 + terms]) - 0.02, 0))
        return np.concatenate(data)

    rows = sum(2 * len(v.image_points) for v in views)
    sparsity = lil_matrix((rows + len(grid), n + 6 * len(views)), dtype=int)
    start = 0
    for i, v in enumerate(views):
        end = start + 2 * len(v.image_points)
        sparsity[start:end, :n] = 1
        sparsity[start:end, n + 6 * i : n + 6 * i + 6] = 1
        start = end
    sparsity[rows:, 4 : 4 + terms] = 1
    lower = np.r_[
        size * 0.1,
        size * 0.1,
        width * 0.25,
        height * 0.25,
        [-2] * terms,
        [-0.03] * 2 if asymmetric else [],
        np.full(6 * len(views), -np.inf),
    ]
    upper = np.r_[
        max(width, height) * 1.2,
        max(width, height) * 1.2,
        width * 0.75,
        height * 0.75,
        [2] * terms,
        [0.03] * 2 if asymmetric else [],
        np.full(6 * len(views), np.inf),
    ]
    fit = least_squares(
        residual,
        np.r_[initial, np.asarray(poses).ravel()],
        bounds=(lower, upper),
        jac_sparsity=sparsity.tocsr(),
        x_scale="jac",
        loss="soft_l1",
        f_scale=2,
        max_nfev=max_nfev,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        tr_options={"atol": 1e-10, "btol": 1e-10},
    )
    m = model(fit.x[:n])
    m.check_domain()
    for i, v in enumerate(views):
        m.project(v.object_points, fit.x[n + 6 * i : n + 6 * i + 6])
    information = []
    start = 0
    for i, v in enumerate(views):
        end = start + 2 * len(v.image_points)
        global_j = fit.jac[start:end, :n].toarray()
        pose_j = fit.jac[start:end, n + 6 * i : n + 6 * i + 6].toarray()
        information.append(
            global_j - pose_j @ np.linalg.lstsq(pose_j, global_j, rcond=None)[0]
        )
        start = end
    information = np.concatenate(information)
    norms = np.linalg.norm(information, axis=0)
    singular = np.linalg.svd(information / np.maximum(norms, 1e-12), compute_uv=False)
    ratio = float(singular[-1] / max(singular[0], 1e-12))
    checks = {
        "converged": bool(fit.success),
        "message": fit.message,
        "evaluations": fit.nfev,
        "monotonic": True,
        "inverse_domain_checked": True,
        "at_parameter_bound": bool(np.any(fit.active_mask[:n])),
        "intrinsic_information_ratio": ratio,
        "weakly_constrained": bool(ratio < 1e-4 or np.any(norms < 1e-8)),
    }
    if (
        not checks["converged"]
        or checks["at_parameter_bound"]
        or checks["weakly_constrained"]
    ):
        raise ValueError(f"Angular calibration failed numerical checks: {checks}")
    return fit.x[:n], fit.x[n:].reshape(-1, 6), checks
