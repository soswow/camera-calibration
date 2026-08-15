"""CLI wrapper for undistorting images with saved calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from camera_calibration.undistort import load_calibration, undistort_path


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "undistort",
        help="Undistort an image or folder using saved K/D",
        description=(
            "Undistort an image or a folder of images using intrinsics JSON "
            "or ROS camera_info YAML."
        ),
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Image file or folder of images to undistort",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("output/intrinsics.json"),
        help="Path to intrinsics JSON or ROS camera_info YAML (default: output/intrinsics.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file (for one image) or output folder (for a folder)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help=(
            "Free-scaling parameter for the new camera matrix: "
            "0=crop black borders (default), 1=keep all pixels"
        ),
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    if not args.calibration.is_file():
        print(f"Error: Calibration file not found: {args.calibration}", file=sys.stderr)
        return 2
    if not (0.0 <= args.alpha <= 1.0):
        print("Error: --alpha must be between 0 and 1", file=sys.stderr)
        return 2
    if not args.source.exists():
        print(f"Error: Source not found: {args.source}", file=sys.stderr)
        return 2

    try:
        calibration = load_calibration(args.calibration)
        result = undistort_path(
            source=args.source,
            calibration=calibration,
            output=args.output,
            alpha=args.alpha,
        )
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Undistorted {len(result.written)} image(s)")
    for path in result.written:
        print(f"  {path}")
    if result.failed:
        print(f"Failed ({len(result.failed)}):")
        for name in result.failed:
            print(f"  - {name}")
    return 0
