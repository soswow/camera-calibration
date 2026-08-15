"""CLI wrapper for ChArUco board PDF/PNG generation."""

from __future__ import annotations

import argparse

from camera_calibration.generate_charuco import main as generate_main


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "generate-charuco",
        help="Generate a print-ready ChArUco board PDF or PNG",
        description="Generate a ChArUco board image (OpenCV aruco).",
        add_help=False,
    )
    parser.set_defaults(handler=run, _raw_argv=True)


def run(args: argparse.Namespace) -> int:
    return generate_main(args.subcommand_argv)
