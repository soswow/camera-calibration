# ChArUco board generator

Generate print-ready ChArUco boards as single-page PDFs or tiled multi-page PDFs with crop marks, tile labels, and a minimap for assembly.

```bash
python -m camera_calibration generate-charuco --help
python generate-charuco/generate_charuco.py --help   # same CLI (shim)
```

<table>
  <tr>
    <td><img src="../docs/images/generate_charuco/charuco_pdf_result.jpg" alt="ChArUco PDF output" width="360"></td>
    <td><img src="../docs/images/generate_charuco/charuco_A1_7x10_79p14mm_margin10mm_tileA3_2x2tiles_minimap.png" alt="Tiled output minimap" width="360"></td>
  </tr>
</table>

## Options

| Option | Default / values | Description |
| --- | --- | --- |
| `--squares-x` | `10` (`>= 2`) | Squares along X. |
| `--squares-y` | `7` (`>= 2`) | Squares along Y. Provide both square counts together. |
| `--square-size` | `70` mm (`> 0`) | Exact square side in millimetres. With `--paper`/`--size` and no square counts, packs the maximum number of squares that fit at this size. |
| `--target-square-size` | unset (`> 0`) | Target square side in millimetres. With `--paper`/`--size` and no square counts, picks how many squares fit then resizes them to fill the page. Cannot be used with `--square-size`. |
| `--marker-proportion` | `0.7` (`(0, 1)`) | Marker side length as a fraction of square size. |
| `--dictionary` | `auto` (OpenCV `DICT_*`) | ArUco dictionary. `auto` picks the smallest 4X4 dictionary that has enough marker IDs. |
| `--paper` | unset (`A0`, `A1`, `A2`, `A3`, `A4`, `LETTER`, `LEGAL`, `TABLOID`) | Main board size by paper name. Cannot be combined with `--size`. |
| `--size` | unset (`WIDTHxHEIGHT` mm) | Board area in millimetres, e.g. `480x720`. Squares are packed into this rectangle. `--tile-paper` only slices it for printing. Cannot be combined with `--paper`. |
| `--tile-paper` | unset (same names as `--paper`) | Tile page size for multi-page PDF output. Requires `--paper` or `--size`. Rotates the tile sheet if that covers the main board with fewer pages. |
| `--dpi` | `300` (`> 0`) | Render resolution used to convert millimetres to pixels. |
| `--margin` | `0` mm (`>= 0`) | Inset on each tile page, or around the board on a single-page `--paper`/`--size` run. When greater than 0, a 30% black legend is drawn just below the board on the bottom-left tile only. |
| `--tile-bleed` | `2` mm (`>= 0`, `<= --margin` when tiling) | Overflow past crop marks on tiled pages. |
| `--crop-mark` | `5` mm (`>= 0`) | Crop mark length on tiled pages. |
| `--output` | `auto` (`output/<name>.pdf`, or a `.png`/`.pdf` path) | Output path. A `.png` or `.pdf` extension selects format. |
| `--format` | `pdf` (`pdf`, `png`) | Output format. Tiled output is PDF only. |

Dictionaries: `DICT_4X4_50`, `DICT_4X4_100`, `DICT_4X4_250`, `DICT_4X4_1000` (same counts for `5X5`, `6X6`, `7X7`), plus `DICT_ARUCO_ORIGINAL`, `DICT_ARUCO_MIP_36H12`, `DICT_APRILTAG_16H5`, `DICT_APRILTAG_25H9`, `DICT_APRILTAG_36H10`, `DICT_APRILTAG_36H11`.

`--paper` or `--size` plus `--squares-x`/`--squares-y` (without `--square-size`) computes square size to fill the board (minus margins). Plus `--square-size` packs as many exact-size squares as will fit. Plus `--target-square-size` picks a count from the target, then resizes squares to fill.

## Examples

```bash
python -m camera_calibration generate-charuco \
  --paper A1 \
  --tile-paper A3 \
  --margin 10 \
  --squares-x 7 \
  --squares-y 10
```

Exact 30 mm squares, as many as fit on a 500×700 mm board tiled to A3:

```bash
python -m camera_calibration generate-charuco \
  --size 500x700 \
  --tile-paper A3 \
  --square-size 30 \
  --margin 10
```

Paper-sized board (A3 @ 300 DPI). Square count is derived from a 70 mm target, then square size is adjusted to fill A3:

```bash
python -m camera_calibration generate-charuco \
  --paper A3 \
  --target-square-size 70 \
  --output charuco_A3.pdf
```

## Tiled output (multi-page PDF)

Tiled output requires PDF format.

- `--tile-paper` requires `--paper` or `--size` (the board area). It does not change that area; it only splits it across tile pages.
- Tile cuts snap to square borders when a square would not fully fit on the current page, so that square moves to the next tile instead of being sliced.
- Tile sheets are rotated when landscape needs fewer pages. `--paper A2 --tile-paper A3` is two landscape A3 pages, not one portrait page.
- `--margin` insets each tile page (and the board on a single-page `--paper`/`--size` run). That shrinks the assembled size vs the named board size; use a small margin if you need to stay close to true A2/A1. When margin is set, a 30% black details line is drawn just below the board on the bottom-left tile only.
- `--tile-bleed` must be `<=` margin. Default bleed is 2 mm; default crop mark length is 5 mm.
- A minimap PNG is written next to the PDF with `_minimap.png` suffix.

## Printing notes

- Laser printer preferred, matte paper
- 300 DPI or higher
- Disable any printer scaling (no “fit to page”)
- Print a reference ruler and verify scale
- After printing, measure square size; target error < 0.5 mm. Use that measurement with [`calibrate`](../calibrate/README.md).
- Mount to a flat surface; do not laminate
