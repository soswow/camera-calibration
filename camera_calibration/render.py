"""Output-to-source ray maps for perspective and equirectangular views."""

import cv2
import numpy as np
from .models import model_for


def remap_view(
    image,
    calibration,
    *,
    projection="perspective",
    fov=100.0,
    yaw=0.0,
    pitch=0.0,
    output_size=None,
    basis=None,
):
    if tuple(image.shape[1::-1]) != tuple(calibration.image_size):
        raise ValueError("Source dimensions do not match calibration")
    if projection not in {"perspective", "equirectangular"}:
        raise ValueError("Unknown output projection")
    if not 0 < fov < 175 or not np.isfinite([fov, yaw, pitch]).all():
        raise ValueError(
            "Perspective FOV must be in (0,175) degrees; angles must be finite"
        )
    width, height = output_size or calibration.image_size
    if width < 2 or height < 2:
        raise ValueError("Output dimensions must be at least 2")
    model = model_for(calibration)
    model.check_domain()
    rotation = (
        basis
        if basis is not None
        else cv2.Rodrigues(np.array([0.0, np.deg2rad(yaw), 0.0]))[0]
        @ cv2.Rodrigues(np.array([-np.deg2rad(pitch), 0.0, 0.0]))[0]
    )
    mx = np.empty((height, width), np.float32)
    my = np.empty_like(mx)
    valid = np.empty((height, width), bool)
    focal = width / (2 * np.tan(np.deg2rad(fov) / 2))
    for start in range(0, height, 128):
        xx, yy = np.meshgrid(
            np.arange(width), np.arange(start, min(start + 128, height))
        )
        if projection == "perspective":
            rays = np.stack(
                (
                    (xx - (width - 1) / 2) / focal,
                    (yy - (height - 1) / 2) / focal,
                    np.ones_like(xx),
                ),
                axis=-1,
            )
        else:
            lon = ((xx + 0.5) / width - 0.5) * 2 * np.pi
            lat = ((yy + 0.5) / height - 0.5) * np.pi
            rays = np.stack(
                (np.sin(lon) * np.cos(lat), np.sin(lat), np.cos(lon) * np.cos(lat)),
                axis=-1,
            )
        shape = xx.shape
        pixels, ok = model.project_rays(rays.reshape(-1, 3) @ rotation.T, strict=False)
        ok &= (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] <= image.shape[1] - 1)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] <= image.shape[0] - 1)
        )
        pixels[~ok] = -1
        mx[start : start + shape[0]] = pixels[:, 0].reshape(shape)
        my[start : start + shape[0]] = pixels[:, 1].reshape(shape)
        valid[start : start + shape[0]] = ok.reshape(shape)
    return cv2.remap(
        image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    ), valid
