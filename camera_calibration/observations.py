"""Portable single-image observation caches and immutable validation membership."""

import json
import hashlib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .detection import DetectedView, DetectionSet


def image_hash(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def save_observations(detections, path):
    payload = asdict(detections)
    for view in payload["views"]:
        view["object_points"] = view["object_points"].tolist()
        view["image_points"] = view["image_points"].tolist()
    payload["schema_version"] = 1
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_observations(path):
    data = json.loads(Path(path).read_text())
    if data.pop("schema_version", None) != 1:
        raise ValueError("Unsupported observation schema")
    views = []
    names = set()
    for v in data.pop("views"):
        if v["name"] in names:
            raise ValueError("Duplicate observation name")
        names.add(v["name"])
        v["object_points"] = np.asarray(v["object_points"], dtype=np.float64).reshape(
            -1, 3
        )
        v["image_points"] = np.asarray(v["image_points"], dtype=np.float64).reshape(
            -1, 1, 2
        )
        if (
            len(v["object_points"]) != len(v["image_points"])
            or len(v["image_points"]) < 4
            or not np.isfinite(v["object_points"]).all()
            or not np.isfinite(v["image_points"]).all()
        ):
            raise ValueError("Invalid cached observation points")
        views.append(DetectedView(**v))
    data["image_size"] = tuple(data["image_size"])
    data["pattern_size"] = tuple(data["pattern_size"])
    return DetectionSet(views=views, **data)


def partition(detections, every=5, reserved=None):
    """Group exact/near duplicates before splitting; reserve whole groups."""
    if every not in {0} and every < 2:
        raise ValueError("validation-every must be 0 or at least 2")
    names = {v.name for v in detections.views}
    if reserved is not None and set(reserved) - names:
        raise ValueError(
            f"Reserved images missing detections: {sorted(set(reserved) - names)}"
        )
    groups = []
    for v in detections.views:
        found = None
        points = {
            tuple(o): p for o, p in zip(v.object_points, v.image_points.reshape(-1, 2))
        }
        for group in groups:
            old = group[0]
            if v.source_sha256 and v.source_sha256 == old.source_sha256:
                found = group
                break
            prev = {
                tuple(o): p
                for o, p in zip(old.object_points, old.image_points.reshape(-1, 2))
            }
            common = points.keys() & prev.keys()
            if (
                len(common) >= 6
                and np.percentile(
                    [np.linalg.norm(points[k] - prev[k]) for k in common], 90
                )
                < 1.0
            ):
                found = group
                break
        if found is None:
            groups.append([v])
        else:
            found.append(v)
    train = []
    valid = []
    duplicates = []
    for i, group in enumerate(groups):
        held = (
            any(v.name in reserved for v in group)
            if reserved is not None
            else bool(every and (i + 1) % every == 0)
        )
        representative = next(
            (v for v in group if reserved is not None and v.name in reserved), group[0]
        )
        (valid if held else train).append(representative)
        duplicates.extend(v.name for v in group if v is not representative)
    return replace(detections, views=train), valid, duplicates


def fit_partitioned(
    detections,
    *,
    model="simple",
    validation_every=5,
    reserved=None,
    auto_select=False,
    auto_select_max_keep=None,
    auto_select_error_factor=1.5,
    auto_select_error_floor=2.0,
    max_rms=None,
    **fit_options,
):
    from .detection import calibrate_detections
    from .validate import validate_view

    train, valid, duplicates = partition(detections, validation_every, reserved)
    if (validation_every or reserved is not None) and len(valid) < 2:
        raise ValueError(
            "Need at least two distinct validation views; add images or explicitly disable validation with --validation-every 0"
        )
    fit = calibrate_detections(
        train,
        distortion_model=model,
        auto_select=auto_select,
        auto_select_max_keep=auto_select_max_keep,
        auto_select_error_factor=auto_select_error_factor,
        auto_select_error_floor=auto_select_error_floor,
        **fit_options,
    )
    records = []
    errors = []
    for view in valid:
        record, e, _ = validate_view(view, fit)
        records.append(asdict(record))
        errors.extend(e.tolist())
    fit.validation = {
        "images": records,
        "rms_px": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        "corners": len(errors),
        "excluded_duplicates": duplicates,
        "reserved_images": [v.name for v in valid],
    }
    import cv2
    import scipy

    fit.provenance = {
        "software": {
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "fit_options": fit_options,
        "max_rms": max_rms,
        "auto_select": auto_select,
        "sources": {v.name: v.source_sha256 for v in detections.views},
        "training_images": fit.used_images,
        "masks": {
            v.name: (v.detection_details or {}).get("exclusion_polygons", [])
            for v in detections.views
        },
        "validation_images": [v.name for v in valid],
        "split_groups": "exact hashes or 90th percentile displacement < 1px across >=6 common target points",
    }
    if max_rms is not None:
        if not np.isfinite(max_rms) or max_rms <= 0:
            raise ValueError("max-rms must be finite and positive")
        if fit.rms_reprojection_error > max_rms or (
            errors and fit.validation["rms_px"] > max_rms
        ):
            raise ValueError(
                f"Fit exceeds max RMS {max_rms}px: training={fit.rms_reprojection_error}, validation={fit.validation['rms_px']}"
            )
    return fit
