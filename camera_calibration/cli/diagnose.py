"""CLI wrapper for calibration coverage / quality diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from camera_calibration.cli.common import add_board_arguments
from camera_calibration.diagnose import diagnose_calibration, render_diagnosis_image
from camera_calibration.result import CalibrationResult


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "diagnose",
        help="Report coverage gaps and per-image reprojection error",
        description=(
            "Diagnose what is missing from a calibration set: "
            "frame coverage, tilt/distance variety, and per-image error outliers."
        ),
    )
    parser.add_argument(
        "images",
        type=Path,
        help="Folder with the calibration images",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="intrinsics JSON or ROS camera_info YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/diagnosis.json"),
        help="Where to write the diagnosis JSON (default: output/diagnosis.json)",
    )
    parser.add_argument(
        "--viz",
        type=Path,
        default=Path("output/diagnosis.png"),
        help="Where to write the visual summary PNG (default: output/diagnosis.png)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip writing the visual summary image",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many worst images to list (default: 10)",
    )
    add_board_arguments(
        parser,
        required_board=False,
        defaults_from_calibration=True,
    )
    parser.set_defaults(handler=run)


def format_report(report, top: int) -> str:
    lines = [
        "Calibration diagnosis",
        f"  Images scored:   {len(report.images)} "
        f"(failed detect: {len(report.failed_images)})",
        f"  Image size:      {report.image_size[0]} x {report.image_size[1]}",
        f"  Overall RMS:     {report.overall_rms:.4f} px",
        f"  Mean of means:   {report.summary['mean_of_mean_errors']:.3f} px",
        f"  Median mean:     {report.summary['median_mean_error']:.3f} px",
        "",
        "Coverage snapshot:",
        f"  Quadrants TL/TR/BL/BR/center: "
        f"{report.summary['quadrant_counts']['TL']}/"
        f"{report.summary['quadrant_counts']['TR']}/"
        f"{report.summary['quadrant_counts']['BL']}/"
        f"{report.summary['quadrant_counts']['BR']}/"
        f"{report.summary['quadrant_counts']['center']}",
        f"  Edge centers L/R/T/B: "
        f"{report.summary['edge_centers']['left']}/"
        f"{report.summary['edge_centers']['right']}/"
        f"{report.summary['edge_centers']['top']}/"
        f"{report.summary['edge_centers']['bottom']}",
        f"  Tilt bins <15 / 15-30 / ≥30: "
        f"{report.summary['tilt_bins']['low_<15']}/"
        f"{report.summary['tilt_bins']['med_15_30']}/"
        f"{report.summary['tilt_bins']['high_>=30']}",
        f"  Distance range:  {report.summary['distance_min']:.1f} – "
        f"{report.summary['distance_max']:.1f} "
        f"(ratio {report.summary['distance_ratio']:.2f})",
        f"  Empty spatial bins: {report.summary['spatial_empty_cells']}/"
        f"{report.summary['spatial_total_cells']} "
        f"(empty frame-corners: {report.summary['spatial_empty_corner_cells']}/4)",
        "",
        "Spatial corner hits (top → bottom rows):",
    ]
    for row in report.spatial_grid:
        lines.append("  " + " ".join(f"{count:5d}" for count in row))

    lines.append("")
    lines.append("Gaps / recommendations:")
    if not report.gaps:
        lines.append("  (none flagged — coverage looks balanced)")
    for gap in report.gaps:
        lines.append(f"  [{gap.severity}] {gap.code}: {gap.message}")

    lines.append("")
    lines.append(f"Worst images by mean reprojection error (top {top}):")
    for image in report.images[:top]:
        lines.append(
            f"  {image.mean_error_px:5.2f} px  tilt={image.tilt_deg:5.1f}°  "
            f"dist={image.distance:6.1f}  quad={image.quadrant:6s}  "
            f"span={max(image.span_x, image.span_y):.2f}  {image.name}"
        )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    if not args.images.is_dir():
        print(f"Error: Not a directory: {args.images}", file=sys.stderr)
        return 2
    if not args.calibration.is_file():
        print(f"Error: Calibration file not found: {args.calibration}", file=sys.stderr)
        return 2

    try:
        calibration = CalibrationResult.from_path(args.calibration)
        if args.board is None:
            args.board = calibration.board_type
        report = diagnose_calibration(
            folder=args.images,
            calibration=calibration,
            detect_scale=args.detect_scale,
            board=args.board,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_size=args.square_size,
            marker_proportion=args.marker_proportion,
            dictionary=args.dictionary,
            min_charuco_corners=args.min_charuco_corners,
        )
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    report.save_json(args.output)
    print(format_report(report, top=args.top))
    print(f"\nWrote {args.output}")
    if not args.no_viz:
        viz_path = render_diagnosis_image(report, args.viz)
        print(f"Wrote {viz_path}")
    return 0
