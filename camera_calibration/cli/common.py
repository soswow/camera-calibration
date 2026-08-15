"""Shared CLI argument groups."""

from __future__ import annotations

import argparse

from camera_calibration.result import BOARD_CHARUCO, BOARD_CHECKERBOARD


def add_board_arguments(parser: argparse.ArgumentParser, *, required_board: bool) -> None:
    parser.add_argument(
        "--board",
        choices=(BOARD_CHARUCO, BOARD_CHECKERBOARD),
        required=required_board,
        default=None if required_board else None,
        help="Calibration target: charuco or checkerboard (same --squares-x/--squares-y for both).",
    )
    parser.add_argument(
        "--squares-x",
        type=int,
        default=None,
        help="Squares along X (same count for charuco and checkerboard)",
    )
    parser.add_argument(
        "--squares-y",
        type=int,
        default=None,
        help="Squares along Y (same count for charuco and checkerboard)",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=None,
        help="Physical square edge length (any unit; stay consistent)",
    )
    parser.add_argument(
        "--marker-proportion",
        type=float,
        default=0.7,
        help="ChArUco marker side as a fraction of square size (default: 0.7)",
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default="DICT_4X4_50",
        help="ChArUco ArUco dictionary name (default: DICT_4X4_50)",
    )
    parser.add_argument(
        "--min-charuco-corners",
        type=int,
        default=6,
        help="Minimum interpolated ChArUco corners to accept a view (default: 6)",
    )
    parser.add_argument(
        "--detect-scale",
        type=float,
        default=0.35,
        help="Checkerboard detection downscale for large photos (default: 0.35)",
    )


def add_auto_select_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--auto-select",
        action="store_true",
        help=(
            "Fit on all detections, drop high-residual / hard close-tilt outliers "
            "while keeping pose diversity, then refit"
        ),
    )
    parser.add_argument(
        "--auto-select-max-keep",
        type=int,
        default=None,
        help="Optional cap on views kept after auto-select (default: keep all inliers)",
    )
    parser.add_argument(
        "--auto-select-error-factor",
        type=float,
        default=1.5,
        help="Reject views with mean error > factor * initial RMS (default: 1.5)",
    )
    parser.add_argument(
        "--auto-select-error-floor",
        type=float,
        default=2.0,
        help="Minimum error threshold in px for outlier rejection (default: 2.0)",
    )


def validate_board_args(
    args: argparse.Namespace,
    *,
    require_square_size: bool,
) -> str | None:
    """Return an error message, or None if the board flags are valid."""
    if args.squares_x is None or args.squares_y is None:
        return "--squares-x and --squares-y are required"
    if args.squares_x < 2 or args.squares_y < 2:
        return "--squares-x and --squares-y must be at least 2"
    if args.board == BOARD_CHECKERBOARD and (args.squares_x < 3 or args.squares_y < 3):
        return "checkerboard needs at least 3 squares on each side"
    if args.board == BOARD_CHARUCO:
        if not (0.0 < args.marker_proportion < 1.0):
            return "--marker-proportion must be in (0, 1)"
        if args.min_charuco_corners < 4:
            return "--min-charuco-corners must be at least 4"
    if require_square_size:
        if args.square_size is None or args.square_size <= 0:
            return "--square-size is required and must be positive"
    elif args.square_size is not None and args.square_size <= 0:
        return "--square-size must be positive"
    return None
