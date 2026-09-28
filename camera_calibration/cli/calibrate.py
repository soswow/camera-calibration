"""CLI wrapper for checkerboard / ChArUco camera calibration."""

from __future__ import annotations

import cv2

import argparse
import json
import sys
from pathlib import Path

from camera_calibration.calibrate import calibrate_from_folder
from camera_calibration.cli.common import (
    add_auto_select_arguments,
    add_board_arguments,
    validate_board_args,
)
from camera_calibration.result import BOARD_CHARUCO


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "calibrate",
        help="Estimate intrinsics from a folder of board images",
        description=(
            "Estimate camera intrinsics from a folder of ChArUco or checkerboard images. "
            "Writes JSON and ROS/OpenCV YAML."
        ),
    )
    parser.add_argument(
        "images",
        type=Path,
        help="Folder with calibration images",
    )
    add_board_arguments(parser, required_board=False)
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("output"),
        help="Directory for JSON and YAML (default: output)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Base filename for .json and .yaml (no path). "
            "If omitted and --camera-name is set, uses "
            "<camera-name>-<board>-<model>"
        ),
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default=None,
        help=(
            "camera_name field for ROS YAML. Also used to build --output-name "
            "when that flag is omitted"
        ),
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Optional folder for images with detected corners drawn",
    )
    parser.add_argument(
        "--model",
        choices=("simple", "full", "k1", "angular", "angular-asymmetric"),
        default="simple",
        help=(
            "Distortion model: simple=k1,k2 only (default, stabler corners), "
            "full=k1..k3+tangential, k1=radial k1 only; angular/angular-asymmetric for fisheye lenses"
        ),
    )
    parser.add_argument("--radial-terms", type=int, choices=range(1, 5), default=2)
    parser.add_argument(
        "--seed-fov",
        type=float,
        default=195.0,
        help="Initial angular FOV across the shorter image dimension",
    )
    parser.add_argument(
        "--max-angle",
        type=float,
        default=110.0,
        help="Angular model maximum half-angle in degrees",
    )
    parser.add_argument("--max-evaluations", type=int, default=500)
    parser.add_argument(
        "--max-rms",
        type=float,
        default=2.0,
        help="Reject training/validation RMS above this pixel threshold",
    )
    parser.add_argument(
        "--board-definition",
        type=Path,
        help="Use board geometry from generate-charuco's .board.json (overrides board flags)",
    )
    parser.add_argument(
        "--validation-every",
        type=int,
        default=5,
        help="Reserve every Nth distinct view before fitting; 0 disables",
    )
    parser.add_argument(
        "--validation-list",
        type=Path,
        help="JSON array of reserved image names; overrides periodic splitting",
    )
    parser.add_argument(
        "--observations-in",
        type=Path,
        help="Read portable cached observations instead of detecting images",
    )
    parser.add_argument(
        "--observations-out", type=Path, help="Save reusable detections before fitting"
    )
    parser.add_argument(
        "--pixel-policy", choices=["encoded", "legacy-exif"], default="encoded"
    )
    parser.add_argument(
        "--masks",
        type=Path,
        help="JSON map of image names to normalized exclusion polygons",
    )
    parser.add_argument("--legacy-pattern", action="store_true")
    parser.add_argument("--first-marker-id", type=int, default=0)
    add_auto_select_arguments(parser)
    parser.set_defaults(handler=run)


def format_report(result) -> str:
    if result.is_angular:
        return (
            f"Calibration complete: {result.distortion_model}\n"
            f"Image size: {result.image_size}; training images: {len(result.used_images)}\n"
            f"Training RMS: {result.rms_reprojection_error:.4f} px\n"
            f"Reserved validation RMS: {(result.validation or {}).get('rms_px')} px\n"
            f"fx/fy: {result.fx:.6f}, {result.fy:.6f}; cx/cy: {result.cx:.6f}, {result.cy:.6f}\n"
            f"Angular radial coefficients: {result.distortion_coefficients}\n"
            f"Angular-plane asymmetric coefficients: {result.asymmetric_coefficients}\n"
            f"Model domain: 0–{result.max_angle_deg:g} degrees off-axis (not a measured lens FOV)\n"
            "JSON profile; no Brown YAML conversion. Fit remains provisional outside observed coverage."
        )
    dist = result.distortion_coefficients
    width, height = result.image_size
    center_x = width / 2.0
    center_y = height / 2.0
    offset_x = result.cx - center_x
    offset_y = result.cy - center_y

    if result.board_type == BOARD_CHARUCO:
        pattern_line = (
            f"  Pattern:         ChArUco {result.pattern_size[0]} x {result.pattern_size[1]} "
            f"squares, square={result.square_size}"
        )
        if result.dictionary:
            pattern_line += f", dict={result.dictionary}"
    else:
        pattern_line = (
            f"  Pattern:         checkerboard {result.pattern_size[0]} x {result.pattern_size[1]} "
            f"squares, square={result.square_size}"
        )

    lines = [
        "Camera calibration complete",
        f"  Images used:     {len(result.used_images)} / "
        f"{len(result.used_images) + len(result.failed_images)}",
        f"  Image size:      {result.image_size[0]} x {result.image_size[1]}",
        pattern_line,
        f"  Distortion model:{result.distortion_model}",
        f"  RMS error:       {result.rms_reprojection_error:.4f} px  (Brown–Conrady)",
    ]
    if result.fitzgibbon_rms_reprojection_error is not None:
        lines.append(
            f"  RMS error:       {result.fitzgibbon_rms_reprojection_error:.4f} px  "
            "(Fitzgibbon λ only)"
        )
    if result.initial_rms_reprojection_error is not None:
        lines.append(
            f"  Initial RMS:     {result.initial_rms_reprojection_error:.4f} px "
            "(before auto-select)"
        )
    if result.auto_select_rejected:
        lines.append(
            f"  Auto-select:     dropped {len(result.auto_select_rejected)}, "
            f"kept {len(result.used_images)} "
            f"(threshold {result.auto_select_threshold_px:.2f} px)"
        )
    if result.rotated_images:
        lines.append(
            f"  EXIF-normalized: {len(result.rotated_images)} "
            "image(s) transformed into the calibration frame"
        )
    lines.extend(
        [
            "",
            "Intrinsics (camera matrix K):",
            f"  fx = {result.fx:.4f}",
            f"  fy = {result.fy:.4f}",
            f"  cx = {result.cx:.4f} ({offset_x:+.2f} from center {center_x:.1f})",
            f"  cy = {result.cy:.4f} ({offset_y:+.2f} from center {center_y:.1f})",
            "",
            "Field of view (pinhole, from K):",
            f"  HFOV = {result.hfov_deg:.2f}°",
            f"  VFOV = {result.vfov_deg:.2f}°",
            f"  DFOV = {result.dfov_deg:.2f}°",
            f"  35mm equiv. = {result.focal_length_35mm_equiv:.1f} mm",
            "",
            "Distortion coefficients [k1, k2, p1, p2, k3, ...] (Brown–Conrady):",
            "  " + ", ".join(f"{value:.6f}" for value in dist),
        ]
    )
    if result.fitzgibbon_lambda is not None:
        lines.extend(
            [
                "",
                "Fitzgibbon division model (1-parameter, alongside Brown–Conrady):",
                f"  λ = {result.fitzgibbon_lambda:.8f}",
                f"  RMS (λ only) = {result.fitzgibbon_rms_reprojection_error:.4f} px",
                "  model: x_d = x_u / (1 + λ r_u²)  (normalized camera plane)",
            ]
        )
    if result.auto_select_rejected:
        lines.append("")
        lines.append(f"Auto-select rejected ({len(result.auto_select_rejected)}):")
        for name in result.auto_select_rejected:
            lines.append(f"  - {name}")
    detect_failed = [
        name
        for name in result.failed_images
        if not result.auto_select_rejected or name not in result.auto_select_rejected
    ]
    if detect_failed:
        lines.append("")
        lines.append(f"Failed detection ({len(detect_failed)}):")
        for name in detect_failed:
            lines.append(f"  - {name}")
    return "\n".join(lines)


def _sanitize_filename_token(token: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in token.strip()
    )
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-._") or "camera"


def _output_name_stem(raw_name: str) -> str:
    stem = Path(raw_name).name
    lower = stem.lower()
    for suffix in (".json", ".yaml", ".yml"):
        if lower.endswith(suffix):
            stem = stem[: -len(suffix)]
            lower = stem.lower()
    return stem or "intrinsics"


def resolve_output_paths(
    output_folder: Path,
    output_name: str | None,
    camera_name: str | None,
    board: str,
    model: str,
) -> tuple[Path, Path, str]:
    """Return (json_path, yaml_path, yaml_camera_name)."""
    yaml_camera_name = camera_name or "camera"
    if output_name:
        stem = _output_name_stem(output_name)
    elif camera_name:
        stem = "-".join(
            (
                _sanitize_filename_token(camera_name),
                _sanitize_filename_token(board),
                _sanitize_filename_token(model),
            )
        )
    else:
        stem = "intrinsics"
    folder = output_folder
    return folder / f"{stem}.json", folder / f"{stem}.yaml", yaml_camera_name


def run(args: argparse.Namespace) -> int:
    if not args.observations_in and not args.images.is_dir():
        print(f"Error: Not a directory: {args.images}", file=sys.stderr)
        return 2
    try:
        if args.board_definition:
            definition = json.loads(args.board_definition.read_text())
            args.board = "charuco"
            args.squares_x = definition["squares_x"]
            args.squares_y = definition["squares_y"]
            args.square_size = definition["square_mm"]
            args.marker_proportion = definition["marker_mm"] / args.square_size
            args.dictionary = definition["dictionary"]
            args.legacy_pattern = definition.get("legacy_pattern", False)
            args.first_marker_id = definition.get("first_marker_id", 0)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        print(f"Error: Invalid board definition: {error}", file=sys.stderr)
        return 2
    if not args.observations_in and args.board not in {"charuco", "checkerboard"}:
        print("Error: provide --board or --board-definition", file=sys.stderr)
        return 2
    board_error = validate_board_args(args, require_square_size=True)
    if board_error and not args.observations_in:
        print(f"Error: {board_error}", file=sys.stderr)
        return 2
    if args.auto_select_max_keep is not None and args.auto_select_max_keep < 3:
        print("Error: --auto-select-max-keep must be at least 3", file=sys.stderr)
        return 2

    try:
        from camera_calibration.observations import (
            load_observations,
            fit_partitioned,
            save_observations,
        )
        from camera_calibration.masks import load_masks

        reserved = (
            json.loads(args.validation_list.read_text())
            if args.validation_list
            else None
        )
        if reserved is not None and (
            not isinstance(reserved, list)
            or not all(isinstance(n, str) for n in reserved)
        ):
            raise ValueError("validation-list must contain an array of image names")
        options = dict(
            validation_every=args.validation_every,
            reserved=reserved,
            auto_select=args.auto_select,
            auto_select_max_keep=args.auto_select_max_keep,
            auto_select_error_factor=args.auto_select_error_factor,
            auto_select_error_floor=args.auto_select_error_floor,
            terms=args.radial_terms,
            seed_fov=args.seed_fov,
            max_angle=args.max_angle,
            max_nfev=args.max_evaluations,
            max_rms=args.max_rms,
        )
        if args.observations_in:
            if args.masks:
                raise ValueError(
                    "Masks must be applied during detection, not to cached observations"
                )
            observations = load_observations(args.observations_in)
            result = fit_partitioned(observations, model=args.model, **options)
            if args.observations_out:
                save_observations(observations, args.observations_out)
        else:
            result = calibrate_from_folder(
                folder=args.images,
                board=args.board,
                square_size=args.square_size,
                squares_x=args.squares_x,
                squares_y=args.squares_y,
                detect_scale=args.detect_scale,
                preview_dir=args.preview_dir,
                distortion_model=args.model,
                marker_proportion=args.marker_proportion,
                dictionary=args.dictionary,
                min_charuco_corners=args.min_charuco_corners,
                pixel_policy=args.pixel_policy,
                masks=load_masks(args.masks),
                legacy_pattern=args.legacy_pattern,
                first_marker_id=args.first_marker_id,
                observations_out=args.observations_out,
                **options,
            )
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    json_path, yaml_path, yaml_camera_name = resolve_output_paths(
        args.output_folder,
        args.output_name,
        args.camera_name,
        args.board,
        args.model,
    )
    result.save_json(json_path)
    if not result.is_angular:
        result.save_ros_yaml(yaml_path, camera_name=yaml_camera_name)
    print(format_report(result))
    print(f"\nWrote {json_path}")
    if not result.is_angular:
        print(f"Wrote {yaml_path} (ROS/OpenCV YAML)")
    if args.preview_dir is not None:
        print(f"Wrote corner previews to {args.preview_dir}")
    return 0
