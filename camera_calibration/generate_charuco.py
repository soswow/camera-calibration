#!/usr/bin/env python3
"""Generate a ChArUco board image for printing."""

from __future__ import annotations

import argparse
import os
import math
import sys

import cv2
from cv2 import aruco
import numpy as np

DEFAULT_SQUARES_X = 10
DEFAULT_SQUARES_Y = 7
DEFAULT_SQUARE_SIZE_MM = 70.0
DEFAULT_MARKER_PROPORTION = 0.7
DEFAULT_DICTIONARY = "auto"
PREFERRED_DICTIONARY = "DICT_4X4_50"
DEFAULT_DPI = 300
DEFAULT_MARGIN_MM = 0.0
DEFAULT_OUTPUT = "auto"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_FORMAT = "pdf"
DEFAULT_TILE_BLEED_MM = 2.0
DEFAULT_CROP_MARK_MM = 5.0
DEFAULT_CROP_STROKE_MM = 0.3
DEFAULT_TILE_LABEL_PT = 8.0
DEFAULT_MINIMAP_MAX_PX = 2000
DETAIL_GRAY = 0.3

PAPER_SIZES_MM = {
    "A0": (841.0, 1189.0),
    "A1": (594.0, 841.0),
    "A2": (420.0, 594.0),
    "A3": (297.0, 420.0),
    "A4": (210.0, 297.0),
    "LETTER": (216.0, 279.0),
    "LEGAL": (216.0, 356.0),
    "TABLOID": (279.0, 432.0),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="camera-calibration generate-charuco",
        description="Generate a ChArUco board image (OpenCV aruco).",
    )
    parser.add_argument(
        "--squares-x",
        type=int,
        default=DEFAULT_SQUARES_X,
        help=f"Number of squares along X. Default: {DEFAULT_SQUARES_X}.",
    )
    parser.add_argument(
        "--squares-y",
        type=int,
        default=DEFAULT_SQUARES_Y,
        help=f"Number of squares along Y. Default: {DEFAULT_SQUARES_Y}.",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=DEFAULT_SQUARE_SIZE_MM,
        help=(
            "Exact square side length in millimetres. With --paper/--size and no "
            "square counts, packs as many squares as will fit at this size. "
            f"Default: {DEFAULT_SQUARE_SIZE_MM}."
        ),
    )
    parser.add_argument(
        "--target-square-size",
        type=float,
        default=None,
        help=(
            "Target square side in millimetres. With --paper/--size and no square "
            "counts, picks how many squares fit then resizes them to fill the page. "
            "Cannot be combined with --square-size."
        ),
    )
    parser.add_argument(
        "--marker-proportion",
        type=float,
        default=DEFAULT_MARKER_PROPORTION,
        help=(
            "Marker side length as a 0-1 proportion of square size. "
            f"Default: {DEFAULT_MARKER_PROPORTION}."
        ),
    )
    paper_choices = ", ".join(PAPER_SIZES_MM)
    parser.add_argument(
        "--dictionary",
        default=DEFAULT_DICTIONARY,
        help=(
            "OpenCV aruco dictionary name, or 'auto' to pick the smallest 4X4 "
            f"dictionary that fits the board. Default: {DEFAULT_DICTIONARY}."
        ),
    )
    parser.add_argument(
        "--paper",
        default=None,
        help=(
            f"Main board size by paper name. Available: {paper_choices}. "
            "Use --size for a custom millimetre size."
        ),
    )
    parser.add_argument(
        "--size",
        default=None,
        help=(
            "Main board size in millimetres as WIDTHxHEIGHT (e.g. 500x700). "
            "Alternative to --paper. Required with --tile-paper if --paper is omitted."
        ),
    )
    parser.add_argument(
        "--tile-paper",
        default=None,
        help=(
            "Tile paper size for multipage output. "
            f"Available: {paper_choices}. Requires --paper or --size."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Render DPI used to convert mm to pixels. Default: {DEFAULT_DPI}.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN_MM,
        help=f"Margin size in millimeters. Default: {DEFAULT_MARGIN_MM}.",
    )
    parser.add_argument(
        "--tile-bleed",
        type=float,
        default=DEFAULT_TILE_BLEED_MM,
        help=(
            "Bleed (overflow) in millimeters beyond crop marks for tiled output. "
            f"Default: {DEFAULT_TILE_BLEED_MM}."
        ),
    )
    parser.add_argument(
        "--crop-mark",
        type=float,
        default=DEFAULT_CROP_MARK_MM,
        help=f"Crop mark length in millimeters. Default: {DEFAULT_CROP_MARK_MM}.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output image path. Default: auto (constructed from arguments).",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf"),
        default=DEFAULT_FORMAT,
        help=f"Output format. Default: {DEFAULT_FORMAT}.",
    )
    return parser.parse_args(argv)


def _dictionary_names() -> list[str]:
    return sorted(name for name in dir(aruco) if name.startswith("DICT_"))


def _dictionary_size(dictionary) -> int:
    return int(np.asarray(dictionary.bytesList).shape[0])


def _get_dictionary(name: str):
    if not hasattr(aruco, name):
        available = ", ".join(_dictionary_names())
        raise SystemExit(f"Unknown dictionary '{name}'. Available: {available}")
    return aruco.getPredefinedDictionary(getattr(aruco, name))


def _board_required_marker_count(board) -> int:
    # ChArUco ids are 0..n-1; OpenCV asserts id < dictionary.bytesList.rows when drawing.
    if hasattr(board, "getIds"):
        ids = np.asarray(board.getIds()).reshape(-1)
    elif hasattr(board, "ids"):
        ids = np.asarray(board.ids).reshape(-1)
    else:
        size = board.getChessboardSize()
        return int(size[0] * size[1]) // 2
    if ids.size == 0:
        return 0
    return int(ids.max()) + 1


def _dictionaries_large_enough(needed: int) -> list[str]:
    all_names = _dictionary_names()
    name_set = set(all_names)
    ranked: list[tuple[int, str]] = []
    for name in all_names:
        # OpenCV exposes duplicate AprilTag aliases that differ only by case.
        if name != name.upper() and name.upper() in name_set:
            continue
        size = _dictionary_size(aruco.getPredefinedDictionary(getattr(aruco, name)))
        if size >= needed:
            ranked.append((size, name))
    ranked.sort(key=lambda item: (0 if "4X4" in item[1] else 1, item[0], item[1]))
    return [name for _, name in ranked]


def _select_dictionary(needed: int) -> str:
    suggestions = _dictionaries_large_enough(needed)
    if not suggestions:
        raise SystemExit(
            f"This board needs {needed} unique ArUco markers, "
            "but no available OpenCV dictionary is large enough."
        )
    return suggestions[0]


def _ensure_dictionary_covers_board(dictionary, board, dictionary_name: str) -> None:
    needed = _board_required_marker_count(board)
    available = _dictionary_size(dictionary)
    if needed <= available:
        return
    suggestions = _dictionaries_large_enough(needed)
    hint = ""
    if suggestions:
        hint = f" Try --dictionary {suggestions[0]}."
        extra = [name for name in suggestions[1:4] if name != suggestions[0]]
        if extra:
            hint += f" Other options: {', '.join(extra)}."
    raise SystemExit(
        f"This board needs {needed} unique ArUco markers, but {dictionary_name} "
        f"only has {available}.{hint} "
        "Alternatively use fewer squares, or a larger --square-size."
    )


def _create_board(
    squares_x: int,
    squares_y: int,
    square_size: float,
    marker_size: float,
    dictionary,
):
    if hasattr(aruco, "CharucoBoard"):
        return aruco.CharucoBoard(
            (squares_x, squares_y),
            square_size,
            marker_size,
            dictionary,
        )
    return aruco.CharucoBoard_create(
        squares_x,
        squares_y,
        square_size,
        marker_size,
        dictionary,
    )


def _render_board(board, size: tuple[int, int], margin_px: int, border_bits: int):
    try:
        if hasattr(board, "generateImage"):
            return board.generateImage(size, marginSize=margin_px, borderBits=border_bits)
        return board.draw(size, marginSize=margin_px, borderBits=border_bits)
    except cv2.error as exc:
        message = str(exc)
        if "generateImageMarker" in message or "bytesList" in message:
            raise SystemExit(
                "OpenCV failed to draw a marker because the dictionary is too small "
                "for this board. Use a larger --dictionary or fewer squares."
            ) from exc
        raise


def _mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def _board_pixel_size(
    squares_x: int,
    squares_y: int,
    square_size_mm: float,
    dpi: int,
) -> tuple[int, int]:
    # Per-square rounding so generateImage is never 1px short of the grid.
    square_px = max(1, _mm_to_px(square_size_mm, dpi))
    return squares_x * square_px, squares_y * square_px


def _mm_to_points(mm: float) -> float:
    return mm / 25.4 * 72.0


def _to_pil_image(img):
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "PDF output requires Pillow. Install with: pip install pillow"
        ) from exc

    if img.ndim == 2:
        return Image.fromarray(img, mode="L")
    if img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb, mode="RGB")
    if img.shape[2] == 4:
        rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(rgba, mode="RGBA")
    raise SystemExit("Unsupported image shape for PDF output.")


def _format_board_details(
    *,
    squares_x: int,
    squares_y: int,
    square_size: float,
    marker_proportion: float,
    marker_size: float,
    dictionary_name: str,
    board_width_mm: float,
    board_height_mm: float,
    dpi: int,
    margin_mm: float,
    paper_label: str | None = None,
    tile_label: str | None = None,
    tile_grid: tuple[int, int] | None = None,
    tile_orientation: str | None = None,
) -> str:
    parts = [
        f"ChArUco {squares_x}x{squares_y}",
        f"square {_fmt_mm_floor(square_size)}mm",
        f"marker {_fmt_mm_floor(marker_size)}mm ({_fmt_prop(marker_proportion)})",
        dictionary_name,
        f"board {_fmt_mm_floor(board_width_mm)}x{_fmt_mm_floor(board_height_mm)}mm",
    ]
    if paper_label:
        parts.append(paper_label)
    if tile_label and tile_grid:
        grid = f"{tile_grid[0]}x{tile_grid[1]}"
        if tile_orientation:
            parts.append(f"tile {tile_label} {tile_orientation} {grid}")
        else:
            parts.append(f"tile {tile_label} {grid}")
    parts.append(f"{dpi}dpi")
    parts.append(f"margin {_fmt_mm_floor(margin_mm)}mm")
    return "  |  ".join(parts)


def _outer_tile_details_edge(col: int, row: int, rows: int) -> str | None:
    # One legend, on the assembled board's bottom-left tile.
    if row == rows - 1 and col == 0:
        return "bottom"
    return None


def _annotation_font(size_px: int):
    from PIL import ImageFont

    candidates = (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size_px)
    return ImageFont.load_default()


def _draw_pdf_margin_details(
    pdf,
    text: str,
    *,
    page_w_pt: float,
    page_h_pt: float,
    margin_pt: float,
    board_x0_pt: float | None = None,
    board_x1_pt: float | None = None,
    board_y0_pt: float | None = None,
) -> None:
    # Sit the legend just below the board, left-aligned to the board edge.
    if margin_pt <= 0 or not text:
        return
    x0 = board_x0_pt if board_x0_pt is not None else 0.0
    x1 = board_x1_pt if board_x1_pt is not None else page_w_pt
    board_bottom = board_y0_pt if board_y0_pt is not None else margin_pt
    usable = max(x1 - x0, page_w_pt * 0.5)
    max_font = min(9.0, margin_pt * 0.5)
    min_font = 3.5
    font_pt = max_font
    while font_pt > min_font and pdf.stringWidth(text, "Helvetica", font_pt) > usable:
        font_pt -= 0.25
    pdf.setFont("Helvetica", font_pt)
    pdf.setFillColorRGB(DETAIL_GRAY, DETAIL_GRAY, DETAIL_GRAY)
    gap = max(1.0, font_pt * 0.25)
    y = board_bottom - gap - font_pt
    if y < 1.0:
        y = 1.0
    pdf.drawString(x0, y, text)


def _draw_raster_margin_details(img, text: str, margin_px: int):
    from PIL import ImageDraw

    original_gray = img.ndim == 2
    pil_img = _to_pil_image(img)
    if pil_img.mode == "L":
        pil_img = pil_img.convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    font_px = max(8, int(round(margin_px * 0.45)))
    max_w = max(8, pil_img.width - 16)
    font = _annotation_font(font_px)
    while font_px > 8:
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_w:
            break
        font_px -= 1
        font = _annotation_font(font_px)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (pil_img.width - text_w) / 2.0
    y = max(0.0, (margin_px - text_h) / 2.0 - bbox[1])
    gray = int(round(DETAIL_GRAY * 255))
    draw.text((x, y), text, font=font, fill=(gray, gray, gray))
    rgb = np.array(pil_img)
    if original_gray:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _write_png(output_path: str, img, details_text: str | None = None, margin_px: int = 0) -> None:
    if details_text and margin_px > 0:
        img = _draw_raster_margin_details(img, details_text, margin_px)
    ok = cv2.imwrite(output_path, img)
    if not ok:
        raise SystemExit(f"Failed to write output image: {output_path}")


def _write_pdf(
    output_path: str,
    img,
    width_mm: float,
    height_mm: float,
    details_text: str | None = None,
    margin_mm: float = 0.0,
) -> None:
    try:
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise SystemExit(
            "PDF output requires reportlab. Install with: pip install reportlab"
        ) from exc

    pil_img = _to_pil_image(img)
    width_pt = _mm_to_points(width_mm)
    height_pt = _mm_to_points(height_mm)
    pdf = canvas.Canvas(output_path, pagesize=(width_pt, height_pt))
    pdf.drawImage(
        ImageReader(pil_img),
        0,
        0,
        width=width_pt,
        height=height_pt,
        preserveAspectRatio=False,
        mask="auto",
    )
    if details_text and margin_mm > 0:
        _draw_pdf_margin_details(
            pdf,
            details_text,
            page_w_pt=width_pt,
            page_h_pt=height_pt,
            margin_pt=_mm_to_points(margin_mm),
            board_x0_pt=_mm_to_points(margin_mm),
            board_x1_pt=width_pt - _mm_to_points(margin_mm),
            board_y0_pt=_mm_to_points(margin_mm),
        )
    pdf.showPage()
    pdf.save()


def _px_to_mm(px: int, dpi: int) -> float:
    return px * 25.4 / dpi


def _square_snapped_spans_mm(
    total_mm: float,
    printable_mm: float,
    square_mm: float,
    origin_mm: float = 0.0,
) -> list[float]:
    # Prefer cuts on square borders so a square that does not fully fit moves
    # to the next tile instead of being sliced. Slice through a square only if
    # it is larger than the printable span.
    if total_mm <= 0 or printable_mm <= 0:
        return [max(total_mm, 0.0)]
    eps = 1e-6
    borders = [0.0, total_mm]
    if square_mm > 0:
        index = 0
        while True:
            border = origin_mm + index * square_mm
            if border > total_mm + eps:
                break
            if border > eps:
                borders.append(min(border, total_mm))
            index += 1
            if index > 10000:
                break
    unique_borders = sorted({round(border, 6) for border in borders})

    spans: list[float] = []
    position = 0.0
    while position < total_mm - eps:
        limit = min(total_mm, position + printable_mm)
        reachable = [
            border
            for border in unique_borders
            if border > position + eps and border <= limit + eps
        ]
        if reachable:
            next_position = max(reachable)
        else:
            next_position = limit
        if next_position <= position + eps:
            next_position = min(total_mm, position + printable_mm)
        spans.append(next_position - position)
        position = next_position
    return spans or [total_mm]


def _spans_mm_to_px(spans_mm: list[float], total_px: int, dpi: int) -> list[int]:
    if not spans_mm:
        return [total_px] if total_px > 0 else []
    spans_px = [max(0, _mm_to_px(span, dpi)) for span in spans_mm]
    spans_px[-1] += total_px - sum(spans_px)
    if spans_px[-1] <= 0 and len(spans_px) > 1:
        spans_px[-2] += spans_px[-1]
        spans_px.pop()
    return [span for span in spans_px if span > 0]


def _choose_tile_layout(
    main_width_mm: float,
    main_height_mm: float,
    tile_width_mm: float,
    tile_height_mm: float,
    margin_mm: float,
    square_size_mm: float,
    origin_x_mm: float = 0.0,
    origin_y_mm: float = 0.0,
) -> tuple[float, float, list[float], list[float]]:
    # Cover the board with printable tile area, snapping joins to square borders.
    candidates: list[tuple[int, int, float, float, list[float], list[float]]] = []
    seen: set[tuple[float, float]] = set()
    for index, (orient_w, orient_h) in enumerate(
        ((tile_width_mm, tile_height_mm), (tile_height_mm, tile_width_mm))
    ):
        key = (orient_w, orient_h)
        if key in seen:
            continue
        seen.add(key)
        printable_w = orient_w - 2 * margin_mm
        printable_h = orient_h - 2 * margin_mm
        if printable_w <= 0 or printable_h <= 0:
            continue
        x_spans = _square_snapped_spans_mm(
            main_width_mm, printable_w, square_size_mm, origin_x_mm
        )
        y_spans = _square_snapped_spans_mm(
            main_height_mm, printable_h, square_size_mm, origin_y_mm
        )
        candidates.append(
            (len(x_spans) * len(y_spans), index, orient_w, orient_h, x_spans, y_spans)
        )
    if not candidates:
        raise SystemExit("margin is too large for the selected tile paper size.")
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, orient_w, orient_h, x_spans, y_spans = candidates[0]
    return orient_w, orient_h, x_spans, y_spans


def _draw_crop_marks(
    pdf,
    trim_x0: float,
    trim_y0: float,
    trim_x1: float,
    trim_y1: float,
    mark_len_pt: float,
    stroke_pt: float,
) -> None:
    if mark_len_pt <= 0 or stroke_pt <= 0:
        return
    dash_len = max(1.0, stroke_pt * 2.0)
    offset = stroke_pt / 2.0

    def draw_dashed(x0: float, y0: float, x1: float, y1: float) -> None:
        pdf.setLineWidth(stroke_pt)
        horizontal = abs(y1 - y0) < 1e-6
        if horizontal:
            start = x0
            end = x1
            step = dash_len if end >= start else -dash_len
            idx = 0
            pos = start
            while (pos <= end if step > 0 else pos >= end):
                next_pos = pos + step
                if step > 0:
                    seg_end = min(next_pos, end)
                else:
                    seg_end = max(next_pos, end)
                if idx % 2 == 0:
                    pdf.setStrokeColorRGB(1, 1, 1)
                else:
                    pdf.setStrokeColorRGB(0, 0, 0)
                pdf.line(pos, y0, seg_end, y0)
                pos = next_pos
                idx += 1
        else:
            start = y0
            end = y1
            step = dash_len if end >= start else -dash_len
            idx = 0
            pos = start
            while (pos <= end if step > 0 else pos >= end):
                next_pos = pos + step
                if step > 0:
                    seg_end = min(next_pos, end)
                else:
                    seg_end = max(next_pos, end)
                if idx % 2 == 0:
                    pdf.setStrokeColorRGB(1, 1, 1)
                else:
                    pdf.setStrokeColorRGB(0, 0, 0)
                pdf.line(x0, pos, x0, seg_end)
                pos = next_pos
                idx += 1

    # Bottom-left
    draw_dashed(
        trim_x0 - mark_len_pt,
        trim_y0 - offset,
        trim_x0,
        trim_y0 - offset,
    )
    draw_dashed(
        trim_x0 - offset,
        trim_y0 - mark_len_pt,
        trim_x0 - offset,
        trim_y0,
    )
    # Bottom-right
    draw_dashed(
        trim_x1,
        trim_y0 - offset,
        trim_x1 + mark_len_pt,
        trim_y0 - offset,
    )
    draw_dashed(
        trim_x1 + offset,
        trim_y0 - mark_len_pt,
        trim_x1 + offset,
        trim_y0,
    )
    # Top-left
    draw_dashed(
        trim_x0 - mark_len_pt,
        trim_y1 + offset,
        trim_x0,
        trim_y1 + offset,
    )
    draw_dashed(
        trim_x0 - offset,
        trim_y1,
        trim_x0 - offset,
        trim_y1 + mark_len_pt,
    )
    # Top-right
    draw_dashed(
        trim_x1,
        trim_y1 + offset,
        trim_x1 + mark_len_pt,
        trim_y1 + offset,
    )
    draw_dashed(
        trim_x1 + offset,
        trim_y1,
        trim_x1 + offset,
        trim_y1 + mark_len_pt,
    )


def _draw_tile_label(
    pdf,
    label: str,
    *,
    trim_x0: float,
    trim_y1: float,
    page_h_pt: float,
    stroke_pt: float,
    font_pt: float,
) -> None:
    pdf.setFont("Helvetica", font_pt)
    label_width = pdf.stringWidth(label, "Helvetica", font_pt)
    pad = max(1.0, stroke_pt * 2.0)
    x = trim_x0 - pad - label_width
    y = trim_y1 + pad + font_pt
    if x < 0 or y > page_h_pt:
        return
    pdf.setFillColorRGB(DETAIL_GRAY, DETAIL_GRAY, DETAIL_GRAY)
    pdf.drawString(x, y - font_pt, label)


def _write_tiled_pdf(
    output_path: str,
    canvas_img,
    *,
    tile_width_mm: float,
    tile_height_mm: float,
    tile_margin_mm: float,
    tile_bleed_mm: float,
    crop_mark_mm: float,
    dpi: int,
    cols: int,
    rows: int,
    col_spans_px: list[int],
    row_spans_px: list[int],
    details_text: str | None = None,
) -> None:
    try:
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas as pdf_canvas
    except ImportError as exc:
        raise SystemExit(
            "PDF output requires reportlab. Install with: pip install reportlab"
        ) from exc

    bleed_px = _mm_to_px(tile_bleed_mm, dpi)

    tile_w_pt = _mm_to_points(tile_width_mm)
    tile_h_pt = _mm_to_points(tile_height_mm)
    margin_pt = _mm_to_points(tile_margin_mm)
    bleed_pt = _mm_to_points(tile_bleed_mm)
    mark_len_pt = _mm_to_points(crop_mark_mm)
    stroke_pt = _mm_to_points(DEFAULT_CROP_STROKE_MM)
    label_pt = DEFAULT_TILE_LABEL_PT

    canvas_w_px = canvas_img.shape[1]
    canvas_h_px = canvas_img.shape[0]
    x_offsets = [0]
    y_offsets = [0]
    for span in col_spans_px:
        x_offsets.append(x_offsets[-1] + span)
    for span in row_spans_px:
        y_offsets.append(y_offsets[-1] + span)

    pdf = pdf_canvas.Canvas(output_path, pagesize=(tile_w_pt, tile_h_pt))
    for row in range(rows):
        for col in range(cols):
            x0_px = x_offsets[col]
            y0_px = y_offsets[row]
            slice_w_px = col_spans_px[col]
            slice_h_px = row_spans_px[row]
            if slice_w_px <= 0 or slice_h_px <= 0:
                continue
            x1_px = x0_px + slice_w_px
            y1_px = y0_px + slice_h_px
            slice_w_mm = _px_to_mm(slice_w_px, dpi)
            slice_h_mm = _px_to_mm(slice_h_px, dpi)

            tile_img_w_px = slice_w_px + 2 * bleed_px
            tile_img_h_px = slice_h_px + 2 * bleed_px
            tile_img = np.full(
                (tile_img_h_px, tile_img_w_px),
                255,
                dtype=canvas_img.dtype,
            )
            src_x0 = max(0, x0_px - bleed_px)
            src_y0 = max(0, y0_px - bleed_px)
            src_x1 = min(canvas_w_px, x1_px + bleed_px)
            src_y1 = min(canvas_h_px, y1_px + bleed_px)

            dst_x0 = src_x0 - (x0_px - bleed_px)
            dst_y0 = src_y0 - (y0_px - bleed_px)
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            dst_y1 = dst_y0 + (src_y1 - src_y0)

            tile_img[dst_y0:dst_y1, dst_x0:dst_x1] = canvas_img[src_y0:src_y1, src_x0:src_x1]

            # Content is top-left aligned so last-row/column remainders stay on the join edges.
            trim_x0 = margin_pt
            trim_y1 = tile_h_pt - margin_pt
            trim_x1 = trim_x0 + _mm_to_points(slice_w_mm)
            trim_y0 = trim_y1 - _mm_to_points(slice_h_mm)
            draw_w_pt = _mm_to_points(slice_w_mm + 2 * tile_bleed_mm)
            draw_h_pt = _mm_to_points(slice_h_mm + 2 * tile_bleed_mm)
            draw_x_pt = trim_x0 - bleed_pt
            draw_y_pt = trim_y0 - bleed_pt

            pil_tile = _to_pil_image(tile_img)
            pdf.drawImage(
                ImageReader(pil_tile),
                draw_x_pt,
                draw_y_pt,
                width=draw_w_pt,
                height=draw_h_pt,
                preserveAspectRatio=False,
                mask="auto",
            )
            _draw_crop_marks(pdf, trim_x0, trim_y0, trim_x1, trim_y1, mark_len_pt, stroke_pt)
            _draw_tile_label(
                pdf,
                f"r{row}c{col}",
                trim_x0=trim_x0,
                trim_y1=trim_y1,
                page_h_pt=tile_h_pt,
                stroke_pt=stroke_pt,
                font_pt=label_pt,
            )
            details_edge = _outer_tile_details_edge(col, row, rows)
            if details_text and tile_margin_mm > 0 and details_edge:
                _draw_pdf_margin_details(
                    pdf,
                    details_text,
                    page_w_pt=tile_w_pt,
                    page_h_pt=tile_h_pt,
                    margin_pt=margin_pt,
                    board_x0_pt=trim_x0,
                    board_x1_pt=trim_x1,
                    board_y0_pt=trim_y0,
                )
            pdf.showPage()
    pdf.save()


def _write_minimap(
    output_path: str,
    canvas_img,
    *,
    col_spans_px: list[int],
    row_spans_px: list[int],
) -> None:
    canvas_h_px, canvas_w_px = canvas_img.shape[:2]
    max_dim = max(canvas_w_px, canvas_h_px)
    scale = 1.0
    if max_dim > DEFAULT_MINIMAP_MAX_PX:
        scale = DEFAULT_MINIMAP_MAX_PX / max_dim
    new_w = max(1, int(round(canvas_w_px * scale)))
    new_h = max(1, int(round(canvas_h_px * scale)))
    if scale != 1.0:
        resized = cv2.resize(canvas_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = canvas_img.copy()
    minimap = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    overlay = minimap.copy()
    thickness = 1
    line_color = (0, 255, 0)
    cols = len(col_spans_px)
    rows = len(row_spans_px)
    min_span = min(col_spans_px + row_spans_px) if col_spans_px and row_spans_px else 1
    font_scale = max(0.3, min(0.8, min_span * scale / 400.0))
    font = cv2.FONT_HERSHEY_SIMPLEX

    x_offsets = [0]
    y_offsets = [0]
    for span in col_spans_px:
        x_offsets.append(x_offsets[-1] + span)
    for span in row_spans_px:
        y_offsets.append(y_offsets[-1] + span)

    for x_px in x_offsets:
        x = int(round(x_px * scale))
        x = min(max(x, 0), new_w - 1)
        cv2.line(overlay, (x, 0), (x, new_h - 1), line_color, thickness)
    for y_px in y_offsets:
        y = int(round(y_px * scale))
        y = min(max(y, 0), new_h - 1)
        cv2.line(overlay, (0, y), (new_w - 1, y), line_color, thickness)

    minimap = cv2.addWeighted(overlay, 0.5, minimap, 0.5, 0)
    for r in range(rows):
        for c in range(cols):
            label = f"r{r}c{c}"
            text_size, baseline = cv2.getTextSize(label, font, font_scale, 1)
            x = int(round(x_offsets[c] * scale)) + 3
            y = int(round(y_offsets[r] * scale)) + 3 + text_size[1]
            if x + text_size[0] >= new_w or y + baseline >= new_h:
                continue
            cv2.putText(
                minimap,
                label,
                (x, y),
                font,
                font_scale,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
    ok = cv2.imwrite(output_path, minimap)
    if not ok:
        raise SystemExit(f"Failed to write minimap: {output_path}")


def _center_paste(canvas_img, board_img) -> None:
    canvas_h, canvas_w = canvas_img.shape[:2]
    board_h, board_w = board_img.shape[:2]
    offset_x = int(round((canvas_w - board_w) / 2))
    offset_y = int(round((canvas_h - board_h) / 2))

    dst_x0 = max(offset_x, 0)
    dst_y0 = max(offset_y, 0)
    dst_x1 = min(offset_x + board_w, canvas_w)
    dst_y1 = min(offset_y + board_h, canvas_h)

    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return

    src_x0 = max(0, -offset_x)
    src_y0 = max(0, -offset_y)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    canvas_img[dst_y0:dst_y1, dst_x0:dst_x1] = board_img[src_y0:src_y1, src_x0:src_x1]


def _paper_size_mm(name: str) -> tuple[float, float]:
    key = name.upper()
    if key not in PAPER_SIZES_MM:
        available = ", ".join(sorted(PAPER_SIZES_MM))
        raise SystemExit(f"Unknown paper size '{name}'. Available: {available}")
    return PAPER_SIZES_MM[key]


def _parse_size_mm(value: str) -> tuple[float, float]:
    token = value.strip().lower().replace("mm", "").replace("×", "x")
    parts = [part.strip() for part in token.split("x")]
    if len(parts) != 2:
        raise SystemExit("size must be WIDTHxHEIGHT in millimetres (e.g. 500x700).")
    try:
        width_mm = float(parts[0])
        height_mm = float(parts[1])
    except ValueError:
        raise SystemExit("size must be WIDTHxHEIGHT in millimetres (e.g. 500x700).")
    if width_mm <= 0 or height_mm <= 0:
        raise SystemExit("size width and height must be > 0.")
    return width_mm, height_mm


def _fmt_mm(mm: float) -> str:
    if abs(mm - round(mm)) < 1e-6:
        return str(int(round(mm)))
    return f"{mm:.2f}".rstrip("0").rstrip(".")


def _fmt_mm_floor(mm: float, decimals: int = 2) -> str:
    factor = 10**decimals
    value = math.floor(mm * factor) / factor
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def _fmt_prop(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _sanitize_token(token: str) -> str:
    return token.replace(".", "p")


def _minimap_path(output_path: str) -> str:
    base, _ = os.path.splitext(output_path)
    return f"{base}_minimap.png"


def _auto_output_name(
    *,
    args: argparse.Namespace,
    squares_x: int,
    squares_y: int,
    square_size: float,
    paper_label: str | None,
    provided_flags: set[str],
    output_format: str,
    tile_label: str | None,
    tile_grid: tuple[int, int] | None,
    dictionary_name: str,
) -> str:
    parts: list[str] = ["charuco"]
    if paper_label:
        parts.append(paper_label)
    parts.append(f"{squares_x}x{squares_y}")

    extras: list[str] = []
    if (
        square_size != DEFAULT_SQUARE_SIZE_MM
        or "--square-size" in provided_flags
        or "--target-square-size" in provided_flags
        or paper_label is not None
    ):
        extras.append(f"{_sanitize_token(_fmt_mm(square_size))}mm")
    if (
        args.marker_proportion != DEFAULT_MARKER_PROPORTION
        or "--marker-proportion" in provided_flags
    ):
        extras.append(f"m{_sanitize_token(_fmt_prop(args.marker_proportion))}")
    if args.dpi != DEFAULT_DPI or "--dpi" in provided_flags:
        extras.append(f"{args.dpi}dpi")
    if args.margin != DEFAULT_MARGIN_MM or "--margin" in provided_flags:
        extras.append(f"margin{_sanitize_token(_fmt_mm(args.margin))}mm")
    if tile_label:
        extras.append(f"tile{tile_label}")
        if tile_grid:
            extras.append(f"{tile_grid[0]}x{tile_grid[1]}tiles")
    if dictionary_name != PREFERRED_DICTIONARY or "--dictionary" in provided_flags:
        extras.append(dictionary_name.lower())

    parts.extend(extras)
    filename = "_".join(parts) + f".{output_format}"
    return os.path.join(DEFAULT_OUTPUT_DIR, filename)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    argv_tokens = list(argv) if argv is not None else sys.argv[1:]
    provided_flags = set()
    for token in argv_tokens:
        if token.startswith("--"):
            provided_flags.add(token.split("=", 1)[0])
    paper_provided = "--paper" in provided_flags
    size_provided = "--size" in provided_flags
    tile_paper_provided = "--tile-paper" in provided_flags
    squares_x_provided = "--squares-x" in provided_flags
    squares_y_provided = "--squares-y" in provided_flags
    squares_provided = squares_x_provided or squares_y_provided
    square_size_provided = "--square-size" in provided_flags
    target_size_provided = "--target-square-size" in provided_flags
    board_bounds_provided = paper_provided or size_provided

    if squares_x_provided ^ squares_y_provided:
        raise SystemExit("Provide both --squares-x and --squares-y together.")
    if paper_provided and size_provided:
        raise SystemExit("Use either --paper or --size, not both.")
    if tile_paper_provided and not board_bounds_provided:
        raise SystemExit("--tile-paper requires --paper or --size to set the main size.")
    if square_size_provided and target_size_provided:
        raise SystemExit("Use either --square-size (exact) or --target-square-size (fill), not both.")
    if target_size_provided and squares_provided:
        raise SystemExit("--target-square-size cannot be combined with --squares-x/--squares-y.")
    if target_size_provided and not board_bounds_provided:
        raise SystemExit("--target-square-size requires --paper or --size.")

    if not (0.0 < args.marker_proportion < 1.0):
        raise SystemExit("marker-proportion must be in the 0-1 range (exclusive).")
    if square_size_provided and args.square_size <= 0:
        raise SystemExit("square-size must be > 0.")
    if target_size_provided and args.target_square_size <= 0:
        raise SystemExit("target-square-size must be > 0.")
    if not square_size_provided and not target_size_provided and args.square_size <= 0:
        raise SystemExit("square-size must be > 0.")
    if args.margin < 0:
        raise SystemExit("margin must be >= 0.")
    if args.tile_bleed < 0:
        raise SystemExit("tile-bleed must be >= 0.")
    if args.crop_mark < 0:
        raise SystemExit("crop-mark must be >= 0.")
    if tile_paper_provided and args.tile_bleed > args.margin:
        raise SystemExit("tile-bleed must be <= margin for tiled output.")
    if args.dpi <= 0:
        raise SystemExit("dpi must be > 0.")

    tile_label = None
    tile_cols = None
    tile_rows = None
    tile_width_mm = None
    tile_height_mm = None
    tile_printable_w = None
    tile_printable_h = None
    fill_to_paper = False

    if board_bounds_provided:
        if paper_provided:
            paper_width_mm, paper_height_mm = _paper_size_mm(args.paper or "A4")
            paper_label = (args.paper or "A4").upper()
        else:
            paper_width_mm, paper_height_mm = _parse_size_mm(args.size)
            paper_label = (
                f"{_sanitize_token(_fmt_mm(paper_width_mm))}x"
                f"{_sanitize_token(_fmt_mm(paper_height_mm))}mm"
            )
        if tile_paper_provided:
            tile_width_mm, tile_height_mm = _paper_size_mm(args.tile_paper)
            tile_label = args.tile_paper.upper()
            if tile_width_mm - 2 * args.margin <= 0 or tile_height_mm - 2 * args.margin <= 0:
                raise SystemExit("margin is too large for the selected tile paper size.")
            # --size/--paper is the checkerboard area. Tiles only slice that area for printing.
            available_w = paper_width_mm
            available_h = paper_height_mm
        else:
            if size_provided:
                available_w = paper_width_mm
                available_h = paper_height_mm
            else:
                available_w = paper_width_mm - 2 * args.margin
                available_h = paper_height_mm - 2 * args.margin
                if available_w <= 0 or available_h <= 0:
                    raise SystemExit("margin is too large for the selected paper size.")

        if squares_provided:
            squares_x = args.squares_x
            squares_y = args.squares_y
            if squares_x < 2 or squares_y < 2:
                raise SystemExit("squares-x and squares-y must be >= 2.")
            if square_size_provided:
                square_size = args.square_size
                if (
                    squares_x * square_size > available_w + 1e-6
                    or squares_y * square_size > available_h + 1e-6
                ):
                    raise SystemExit(
                        "Exact --square-size with these square counts does not fit "
                        "the printable area."
                    )
            else:
                square_size = min(available_w / squares_x, available_h / squares_y)
                fill_to_paper = True
        elif target_size_provided:
            squares_x = int(round(available_w / args.target_square_size))
            squares_y = int(round(available_h / args.target_square_size))
            if squares_x < 2 or squares_y < 2:
                raise SystemExit(
                    "paper size is too small for the requested target-square-size."
                )
            square_size = min(available_w / squares_x, available_h / squares_y)
            fill_to_paper = True
        else:
            square_size = args.square_size
            squares_x = int(math.floor(available_w / square_size + 1e-9))
            squares_y = int(math.floor(available_h / square_size + 1e-9))
            if squares_x < 2 or squares_y < 2:
                raise SystemExit("paper size is too small for the requested square-size.")

        if tile_paper_provided or size_provided:
            output_width_mm = available_w
            output_height_mm = available_h
        else:
            output_width_mm = paper_width_mm
            output_height_mm = paper_height_mm
    else:
        squares_x = args.squares_x
        squares_y = args.squares_y
        if squares_x < 2 or squares_y < 2:
            raise SystemExit("squares-x and squares-y must be >= 2.")
        square_size = args.square_size
        output_width_mm = squares_x * square_size + 2 * args.margin
        output_height_mm = squares_y * square_size + 2 * args.margin
        paper_label = None

    marker_size = square_size * args.marker_proportion
    board_width_mm = squares_x * square_size
    board_height_mm = squares_y * square_size
    tile_x_spans_mm: list[float] | None = None
    tile_y_spans_mm: list[float] | None = None
    if tile_paper_provided:
        origin_x_mm = max(0.0, (available_w - board_width_mm) / 2.0)
        origin_y_mm = max(0.0, (available_h - board_height_mm) / 2.0)
        tile_width_mm, tile_height_mm, tile_x_spans_mm, tile_y_spans_mm = (
            _choose_tile_layout(
                available_w,
                available_h,
                tile_width_mm,
                tile_height_mm,
                args.margin,
                square_size,
                origin_x_mm,
                origin_y_mm,
            )
        )
        tile_cols = len(tile_x_spans_mm)
        tile_rows = len(tile_y_spans_mm)
        tile_printable_w = tile_width_mm - 2 * args.margin
        tile_printable_h = tile_height_mm - 2 * args.margin

    dictionary_auto = args.dictionary.lower() == DEFAULT_DICTIONARY
    if dictionary_auto:
        needed_estimate = (squares_x * squares_y) // 2
        dictionary_name = _select_dictionary(needed_estimate)
    else:
        dictionary_name = args.dictionary
    dictionary = _get_dictionary(dictionary_name)
    board = _create_board(
        squares_x,
        squares_y,
        square_size,
        marker_size,
        dictionary,
    )
    needed_markers = _board_required_marker_count(board)
    if dictionary_auto and needed_markers > _dictionary_size(dictionary):
        dictionary_name = _select_dictionary(needed_markers)
        dictionary = _get_dictionary(dictionary_name)
        board = _create_board(
            squares_x,
            squares_y,
            square_size,
            marker_size,
            dictionary,
        )
    elif not dictionary_auto:
        _ensure_dictionary_covers_board(dictionary, board, dictionary_name)
    output_path = args.output
    output_format = args.format.lower()
    if output_path != DEFAULT_OUTPUT:
        ext = os.path.splitext(output_path)[1].lower()
        if ext in {".png", ".pdf"}:
            ext_format = ext[1:]
            if "--format" in provided_flags and ext_format != output_format:
                raise SystemExit(
                    f"Output extension ({ext}) does not match --format {output_format}."
                )
            output_format = ext_format
    else:
        output_path = _auto_output_name(
            args=args,
            squares_x=squares_x,
            squares_y=squares_y,
            square_size=square_size,
            paper_label=paper_label,
            provided_flags=provided_flags,
            output_format=output_format,
            tile_label=tile_label,
            tile_grid=(tile_cols, tile_rows) if tile_paper_provided else None,
            dictionary_name=dictionary_name,
        )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    details_text = None
    if args.margin > 0:
        tile_orientation = None
        if tile_paper_provided:
            tile_orientation = (
                "landscape" if tile_width_mm > tile_height_mm else "portrait"
            )
        details_text = _format_board_details(
            squares_x=squares_x,
            squares_y=squares_y,
            square_size=square_size,
            marker_proportion=args.marker_proportion,
            marker_size=marker_size,
            dictionary_name=dictionary_name,
            board_width_mm=board_width_mm,
            board_height_mm=board_height_mm,
            dpi=args.dpi,
            margin_mm=args.margin,
            paper_label=paper_label,
            tile_label=tile_label,
            tile_grid=(tile_cols, tile_rows) if tile_paper_provided else None,
            tile_orientation=tile_orientation,
        )
    minimap_path = None
    if tile_paper_provided:
        if output_format != "pdf":
            raise SystemExit("Tiled output requires PDF format.")
        board_w_px, board_h_px = _board_pixel_size(
            squares_x, squares_y, square_size, args.dpi
        )
        canvas_w_px = max(_mm_to_px(available_w, args.dpi), board_w_px)
        canvas_h_px = max(_mm_to_px(available_h, args.dpi), board_h_px)
        board_img = _render_board(board, (board_w_px, board_h_px), 0, border_bits=1)
        canvas_img = np.full((canvas_h_px, canvas_w_px), 255, dtype=board_img.dtype)
        _center_paste(canvas_img, board_img)
        col_spans_px = _spans_mm_to_px(tile_x_spans_mm, canvas_w_px, args.dpi)
        row_spans_px = _spans_mm_to_px(tile_y_spans_mm, canvas_h_px, args.dpi)
        _write_tiled_pdf(
            output_path,
            canvas_img,
            tile_width_mm=tile_width_mm,
            tile_height_mm=tile_height_mm,
            tile_margin_mm=args.margin,
            tile_bleed_mm=args.tile_bleed,
            crop_mark_mm=args.crop_mark,
            dpi=args.dpi,
            cols=tile_cols,
            rows=tile_rows,
            col_spans_px=col_spans_px,
            row_spans_px=row_spans_px,
            details_text=details_text,
        )
        minimap_path = _minimap_path(output_path)
        _write_minimap(
            minimap_path,
            canvas_img,
            col_spans_px=col_spans_px,
            row_spans_px=row_spans_px,
        )
        width_px = canvas_w_px
        height_px = canvas_h_px
    else:
        width_px = _mm_to_px(output_width_mm, args.dpi)
        height_px = _mm_to_px(output_height_mm, args.dpi)
        margin_px = _mm_to_px(args.margin, args.dpi)
        if board_bounds_provided and not fill_to_paper:
            board_w_px, board_h_px = _board_pixel_size(
                squares_x, squares_y, square_size, args.dpi
            )
            board_img = _render_board(board, (board_w_px, board_h_px), 0, border_bits=1)
            img = np.full((height_px, width_px), 255, dtype=board_img.dtype)
            _center_paste(img, board_img)
        else:
            img = _render_board(board, (width_px, height_px), margin_px, border_bits=1)
        if output_format == "png":
            _write_png(output_path, img, details_text=details_text, margin_px=margin_px)
        elif output_format == "pdf":
            _write_pdf(
                output_path,
                img,
                output_width_mm,
                output_height_mm,
                details_text=details_text,
                margin_mm=args.margin,
            )
        else:
            raise SystemExit(f"Unknown output format: {output_format}")

    print("ChArUco board written:")
    print(f"  output: {output_path}")
    print(f"  squares: {squares_x} x {squares_y}")
    print(f"  square size (mm): {_fmt_mm_floor(square_size)}")
    if (
        target_size_provided
        and abs(square_size - args.target_square_size) > 0.01
    ):
        print(f"  requested square size (mm): {_fmt_mm_floor(args.target_square_size)}")
    print(f"  marker proportion: {args.marker_proportion}")
    print(f"  marker size (mm): {_fmt_mm_floor(marker_size)}")
    print(f"  dictionary: {dictionary_name}" + (" (auto)" if dictionary_auto else ""))
    print(
        "  board size (mm):"
        f" {_fmt_mm_floor(board_width_mm)} x {_fmt_mm_floor(board_height_mm)}"
    )
    if board_bounds_provided:
        size_kind = "paper" if paper_provided else "size"
        if tile_paper_provided:
            print(f"  {size_kind}: {paper_label} (main)")
        else:
            print(f"  {size_kind}: {paper_label}")
    if tile_paper_provided:
        tile_orientation = (
            "landscape" if tile_width_mm > tile_height_mm else "portrait"
        )
        print(f"  tile paper: {tile_label} ({tile_orientation})")
        print(f"  tiles: {tile_cols} x {tile_rows}")
        print(f"  tile margin (mm): {_fmt_mm_floor(args.margin)}")
        print(f"  tile bleed (mm): {_fmt_mm_floor(args.tile_bleed)}")
        print(
            "  tile printable (mm):"
            f" {_fmt_mm_floor(tile_printable_w)} x {_fmt_mm_floor(tile_printable_h)}"
        )
        print(
            "  tiled area (mm):"
            f" {_fmt_mm_floor(output_width_mm)} x {_fmt_mm_floor(output_height_mm)}"
        )
        if minimap_path:
            print(f"  minimap: {minimap_path}")
    else:
        print(
            "  output size (mm):"
            f" {_fmt_mm_floor(output_width_mm)} x {_fmt_mm_floor(output_height_mm)}"
        )
    print(f"  pixels: {width_px} x {height_px}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
