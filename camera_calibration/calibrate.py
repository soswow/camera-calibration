"""Folder calibration entry points for checkerboard and ChArUco boards."""

from __future__ import annotations

from pathlib import Path

from .detection import DetectionSet
from .result import BOARD_CHARUCO, BOARD_CHECKERBOARD, CalibrationResult


def collect_board_detections(
    folder: Path,
    board: str,
    *,
    squares_x: int | None = None,
    squares_y: int | None = None,
    square_size: float,
    detect_scale: float = 0.35,
    preview_dir: Path | None = None,
    marker_proportion: float = 0.7,
    dictionary: str = "DICT_4X4_50",
    min_charuco_corners: int = 6,
    min_views: int = 3,
    pixel_policy="encoded",
    masks=None,
    legacy_pattern=False,
    first_marker_id=0,
) -> DetectionSet:
    """Collect detections for the requested board type."""
    if squares_x is None or squares_y is None:
        raise ValueError("Calibration requires --squares-x and --squares-y")

    if board == BOARD_CHECKERBOARD:
        from .checkerboard import collect_detections as collect_checkerboard

        return collect_checkerboard(
            folder=folder,
            squares_x=squares_x,
            squares_y=squares_y,
            square_size=square_size,
            detect_scale=detect_scale,
            preview_dir=preview_dir,
            min_views=min_views,
            pixel_policy=pixel_policy,
            masks=masks,
        )

    if board == BOARD_CHARUCO:
        from .charuco import collect_detections as collect_charuco

        return collect_charuco(
            folder=folder,
            squares_x=squares_x,
            squares_y=squares_y,
            square_size=square_size,
            marker_proportion=marker_proportion,
            dictionary_name=dictionary,
            legacy_pattern=legacy_pattern,
            first_marker_id=first_marker_id,
            min_corners=min_charuco_corners,
            preview_dir=preview_dir,
            min_views=min_views,
            pixel_policy=pixel_policy,
            masks=masks,
        )

    raise ValueError(f"Unknown board type {board!r}")


def calibrate_from_folder(
    folder: Path,
    board: str,
    square_size: float,
    *,
    squares_x: int | None = None,
    squares_y: int | None = None,
    detect_scale: float = 0.35,
    preview_dir: Path | None = None,
    distortion_model: str = "simple",
    auto_select: bool = False,
    auto_select_max_keep: int | None = None,
    auto_select_error_factor: float = 1.5,
    auto_select_error_floor: float = 2.0,
    marker_proportion: float = 0.7,
    dictionary: str = "DICT_4X4_50",
    min_charuco_corners: int = 6,
    pixel_policy="encoded",
    masks=None,
    legacy_pattern=False,
    first_marker_id=0,
    validation_every=0,
    reserved=None,
    observations_out=None,
    **fit_options,
) -> CalibrationResult:
    """Calibrate a camera from a folder of board images."""
    detections = collect_board_detections(
        folder=folder,
        board=board,
        squares_x=squares_x,
        squares_y=squares_y,
        square_size=square_size,
        detect_scale=detect_scale,
        preview_dir=preview_dir,
        marker_proportion=marker_proportion,
        dictionary=dictionary,
        min_charuco_corners=min_charuco_corners,
        pixel_policy=pixel_policy,
        masks=masks,
        legacy_pattern=legacy_pattern,
        first_marker_id=first_marker_id,
    )
    from .observations import fit_partitioned, save_observations

    if observations_out is not None:
        save_observations(detections, observations_out)
    return fit_partitioned(
        detections,
        model=distortion_model,
        validation_every=validation_every,
        reserved=reserved,
        **fit_options,
        auto_select=auto_select,
        auto_select_max_keep=auto_select_max_keep,
        auto_select_error_factor=auto_select_error_factor,
        auto_select_error_floor=auto_select_error_floor,
    )
