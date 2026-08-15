"""CLI wrapper for visualizing saved camera intrinsics / distortion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from camera_calibration.result import CalibrationResult
from camera_calibration.visualize import render_distortion_figure, render_undistort_comparison


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "visualize",
        help="Plot K / distortion from JSON or ROS YAML",
        description=(
            "Visualize camera matrix and distortion from a calibration JSON or "
            "ROS camera_info YAML: warped grid, undistort displacement, and "
            "radial curve."
        ),
    )
    parser.add_argument(
        "calibration",
        type=Path,
        help="intrinsics JSON or ROS/OpenCV camera_info YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG for the 4-panel figure (default: <calibration-stem>-viz.png)",
    )
    parser.add_argument(
        "--exaggerate",
        type=float,
        default=1.0,
        help=(
            "Scale the warped-grid offset only (heatmap/curves stay true). "
            "Use e.g. 10 when k1 is tiny and the red grid looks straight (default: 1)"
        ),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional photo at the calibrated resolution for a before/after PNG",
    )
    parser.add_argument(
        "--compare-output",
        type=Path,
        default=None,
        help="Where to write the before/after PNG (default: <stem>-undistort-compare.png)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="cv2.getOptimalNewCameraMatrix alpha for --image (default: 0 = crop)",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    if not args.calibration.is_file():
        print(f"Error: Calibration file not found: {args.calibration}", file=sys.stderr)
        return 2
    if args.exaggerate <= 0:
        print("Error: --exaggerate must be positive", file=sys.stderr)
        return 2
    if not (0.0 <= args.alpha <= 1.0):
        print("Error: --alpha must be between 0 and 1", file=sys.stderr)
        return 2
    if args.image is not None and not args.image.is_file():
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        return 2

    output = args.output
    if output is None:
        output = args.calibration.with_name(f"{args.calibration.stem}-viz.png")

    try:
        calibration = CalibrationResult.from_path(args.calibration)
        viz_path = render_distortion_figure(
            calibration,
            output,
            exaggerate=args.exaggerate,
        )
        print(f"Wrote {viz_path}")
        if args.image is not None:
            compare_output = args.compare_output
            if compare_output is None:
                compare_output = args.calibration.with_name(
                    f"{args.calibration.stem}-undistort-compare.png"
                )
            compare_path = render_undistort_comparison(
                calibration,
                args.image,
                compare_output,
                alpha=args.alpha,
            )
            print(f"Wrote {compare_path}")
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
