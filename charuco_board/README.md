# ChArUco Board Generator

Generate print-ready ChArUco boards as single-page PDFs or tiled multi-page PDFs with crop marks, tile labels, and a minimap for assembly.

<table>
  <tr>
    <td><img src="../docs/images/generate_charuco/charuco_pdf_result.jpg" alt="ChArUco PDF output" width="360"></td>
    <td><img src="../docs/images/generate_charuco/charuco_A1_7x10_79p14mm_margin10mm_tileA3_2x2tiles_minimap.png" alt="Tiled output minimap" width="360"></td>
  </tr>
</table>

## Usage

Run from the repository root.

Options:

| Option | Default / values | Description |
| --- | --- | --- |
| `--squares-x` | `10` (`>= 2`) | Squares along X. |
| `--squares-y` | `7` (`>= 2`) | Squares along Y. Provide both square counts together. |
| `--square-size` | `70` mm (`> 0`) | Exact square side in millimetres. With `--paper`/`--size` and no square counts, packs the maximum number of squares that fit at this size. |
| `--target-square-size` | unset (`> 0`) | Target square side in millimetres. With `--paper`/`--size` and no square counts, picks how many squares fit then resizes them to fill the page. Cannot be used with `--square-size`. |
| `--marker-proportion` | `0.7` (`(0, 1)`) | Marker side length as a fraction of square size. |
| `--dictionary` | `auto` (OpenCV `DICT_*`) | ArUco dictionary. `auto` picks the smallest 4X4 dictionary that has enough marker IDs. |
| `--paper` | unset (`A0`, `A1`, `A2`, `A3`, `A4`, `LETTER`, `LEGAL`, `TABLOID`) | Main board size by paper name. Cannot be combined with `--size`. |
| `--size` | unset (`WIDTHxHEIGHT` mm) | Main board size in millimetres, e.g. `500x700`. Alternative to `--paper`. |
| `--tile-paper` | unset (same names as `--paper`) | Tile page size for multi-page PDF output. Requires `--paper` or `--size`. Rotates the tile sheet if that covers the main board with fewer pages. |
| `--dpi` | `300` (`> 0`) | Render resolution used to convert millimetres to pixels. |
| `--margin` | `0` mm (`>= 0`) | Inset on each tile page, or around the board on a single-page `--paper`/`--size` run. When greater than 0, a 30% black legend of board details is drawn along an outer margin (not on edges that are cut away between tiles). |
| `--tile-bleed` | `2` mm (`>= 0`, `<= --margin` when tiling) | Overflow past crop marks on tiled pages. |
| `--crop-mark` | `5` mm (`>= 0`) | Crop mark length on tiled pages. |
| `--output` | `auto` (`output/<name>.pdf`, or a `.png`/`.pdf` path) | Output path. A `.png` or `.pdf` extension selects format. |
| `--format` | `pdf` (`pdf`, `png`) | Output format. Tiled output is PDF only. |

Dictionaries: `DICT_4X4_50`, `DICT_4X4_100`, `DICT_4X4_250`, `DICT_4X4_1000` (same counts for `5X5`, `6X6`, `7X7`), plus `DICT_ARUCO_ORIGINAL`, `DICT_ARUCO_MIP_36H12`, `DICT_APRILTAG_16H5`, `DICT_APRILTAG_25H9`, `DICT_APRILTAG_36H10`, `DICT_APRILTAG_36H11`.

`--paper` or `--size` plus `--squares-x`/`--squares-y` (without `--square-size`) computes square size to fill the board (minus margins). Plus `--square-size` packs as many exact-size squares as will fit. Plus `--target-square-size` picks a count from the target, then resizes squares to fill.

```bash
python charuco_board/generate_charuco.py \
  --paper A1 \
  --tile-paper A3 \
  --margin 10 \
  --squares-x 7 \
  --squares-y 10
```

Exact 30 mm squares, as many as fit on a 500×700 mm board tiled to A3:

```bash
python charuco_board/generate_charuco.py \
  --size 500x700 \
  --tile-paper A3 \
  --square-size 30 \
  --margin 10
```

Example for a paper-sized board (A3 @ 300 DPI). Square count is derived from a 70 mm target, then square size is adjusted to fill A3:

```bash
python charuco_board/generate_charuco.py \
  --paper A3 \
  --target-square-size 70 \
  --output charuco_A3.pdf
```

## Tiled output (multi-page PDF)

Tiled output requires PDF format.

```bash
python charuco_board/generate_charuco.py \
  --paper A1 \
  --tile-paper A3 \
  --margin 10 \
  --squares-x 8 \
  --squares-y 12
```

Tiling notes:
- `--tile-paper` requires `--paper` or `--size` (the main board size).
- Tile sheets are rotated when landscape needs fewer pages. `--paper A2 --tile-paper A3` is two landscape A3 pages, not one portrait page.
- `--margin` insets each tile page (and the board on a single-page `--paper`/`--size` run). That shrinks the assembled size vs the named board size; use a small margin if you need to stay close to true A2/A1. When margin is set, a 30% black details line is drawn on an outer margin, not on joins between tiles.
- `--tile-bleed` controls the overflow beyond crop marks (default: 2 mm). Bleed must be `<=` margin.
- `--crop-mark` controls crop mark length (default: 5 mm).
- A minimap PNG is written next to the PDF with `_minimap.png` suffix.

## Printing notes

- Laser printer preferred, matte paper
- 300 DPI or higher
- Disable any printer scaling (no “fit to page”)
- Print a reference ruler and verify scale
- After printing, measure square size; target error < 0.5 mm
- Mount to a flat surface; do not laminate
