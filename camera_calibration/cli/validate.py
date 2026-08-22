"""CLI for held-out calibration validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from camera_calibration.cli.common import add_board_arguments
from camera_calibration.result import CalibrationResult
from camera_calibration.validate import validate_calibration


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "validate",
        help="Measure held-out reprojection RMS and grid straightness",
        description=(
            "Validate saved intrinsics on one image or a folder. Intrinsics stay "
            "fixed; only each board pose is estimated."
        ),
    )
    parser.add_argument("images", type=Path, help="Validation image or folder")
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Intrinsics JSON or ROS camera_info YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/validation.json"),
        help="Output report JSON (default: output/validation.json)",
    )
    parser.add_argument(
        "--viz",
        type=Path,
        default=None,
        help=(
            "Overlay PNG for one image, or output folder for an image folder "
            "(default: derived from --output)"
        ),
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Do not write undistorted grid-line overlays",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Undistortion framing: 0=crop black borders, 1=keep all pixels",
    )
    parser.add_argument(
        "--line-opacity",
        type=float,
        default=0.55,
        help="Fitted grid-line opacity from 0 to 1 (default: 0.55)",
    )
    add_board_arguments(
        parser,
        required_board=False,
        defaults_from_calibration=True,
    )
    parser.set_defaults(handler=run)


def _format_metrics(label: str, metrics) -> str:
    return (
        f"  {label:<14} RMS {metrics.rms_px:7.3f} px   "
        f"mean {metrics.mean_px:7.3f}   median {metrics.median_px:7.3f}   "
        f"p95 {metrics.p95_px:7.3f}   max {metrics.max_px:7.3f}"
    )


def format_report(report) -> str:
    metrics = report.validation_reprojection
    lines = [
        "Calibration validation",
        f"  Images scored:  {len(report.images)} (failed: {len(report.failed_images)})",
        f"  Corners scored: {sum(image.corner_count for image in report.images)}",
        f"  Model:          {report.distortion_model}",
        f"  Training RMS:   {report.calibration_training_rms_px:.4f} px (reference only)",
        _format_metrics("Validation", metrics),
    ]
    if report.validation_straightness is not None:
        straight = report.validation_straightness
        lines.append(
            f"  {'Straightness':<14} RMS {straight.rms_px:7.3f} px   "
            f"median {straight.median_px:7.3f}   p95 {straight.p95_px:7.3f}   "
            f"max {straight.max_px:7.3f}   "
            f"lines {straight.row_lines} row/{straight.column_lines} column"
        )

    lines.extend(["", "Per-image validation:"])
    for image in sorted(report.images, key=lambda item: -item.reprojection.rms_px):
        marker = "  [NOT HELD OUT]" if image.name in report.overlap_images else ""
        lines.append(
            f"  RMS {image.reprojection.rms_px:7.3f} px  "
            f"p95 {image.reprojection.p95_px:7.3f}  "
            f"corners {image.corner_count:4d}  {image.name}{marker}"
        )
        if image.visualization:
            lines.append(f"        visualization: {image.visualization}")

    if report.failed_images:
        lines.extend(["", "Failed images:"])
        for name, reason in report.failed_images.items():
            lines.append(f"  {name}: {reason}")
    if report.overlap_images:
        lines.extend(
            [
                "",
                "WARNING: Some images participated in calibration or auto-selection.",
                "Their scores are diagnostics, not independent validation.",
            ]
        )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    if not args.images.exists():
        print(f"Error: Image or folder not found: {args.images}", file=sys.stderr)
        return 2
    if not args.calibration.is_file():
        print(f"Error: Calibration file not found: {args.calibration}", file=sys.stderr)
        return 2
    if not (0.0 <= args.alpha <= 1.0):
        print("Error: --alpha must be between 0 and 1", file=sys.stderr)
        return 2
    if not (0.0 <= args.line_opacity <= 1.0):
        print("Error: --line-opacity must be between 0 and 1", file=sys.stderr)
        return 2

    visualization_output = None
    if not args.no_viz:
        if args.viz is not None:
            visualization_output = args.viz
        elif args.images.is_file():
            visualization_output = args.output.with_name(f"{args.output.stem}-viz.png")
        else:
            visualization_output = args.output.with_name(f"{args.output.stem}-viz")

    try:
        calibration = CalibrationResult.from_path(args.calibration)
        report = validate_calibration(
            args.images,
            calibration,
            calibration_file=args.calibration,
            board_type=args.board,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_size=args.square_size,
            marker_proportion=args.marker_proportion,
            dictionary=args.dictionary,
            min_charuco_corners=args.min_charuco_corners,
            detect_scale=args.detect_scale,
            visualization_output=visualization_output,
            visualization_alpha=args.alpha,
            line_opacity=args.line_opacity,
        )
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    report.save_json(args.output)
    print(format_report(report))
    print(f"\nWrote {args.output}")
    return 0
