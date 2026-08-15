"""Unified CLI: python -m camera_calibration <command>."""

from __future__ import annotations

import argparse
import sys

from . import calibrate, diagnose, generate_charuco, undistort, visualize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="camera-calibration",
        description=(
            "Camera calibration helpers: generate ChArUco boards, calibrate from "
            "ChArUco or checkerboard photos, undistort, visualize, and diagnose."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_charuco.add_parser(subparsers)
    calibrate.add_parser(subparsers)
    undistort.add_parser(subparsers)
    visualize.add_parser(subparsers)
    diagnose.add_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if argv_list[:1] == ["generate-charuco"]:
        # Forward remaining flags to the board generator's own argparse.
        return generate_charuco.generate_main(argv_list[1:])

    parser = build_parser()
    args = parser.parse_args(argv_list)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    return handler(args)
