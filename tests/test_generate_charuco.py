from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from reportlab.pdfbase.pdfmetrics import stringWidth

from camera_calibration.generate_charuco import _draw_pdf_margin_details, _mm_to_points, main


class MarginDetailsTests(unittest.TestCase):
    def draw_details(self, margin_mm: float, text: str):
        pdf = Mock()
        pdf.stringWidth.side_effect = stringWidth
        _draw_pdf_margin_details(
            pdf, text, page_w_pt=_mm_to_points(297),
            page_h_pt=_mm_to_points(420), margin_pt=_mm_to_points(margin_mm),
        )
        return pdf

    def test_two_mm_margin_enforces_minimum_font_size(self):
        pdf = self.draw_details(2, "ChArUco 5x5 | square 30mm")
        pdf.setFont.assert_called_once_with("Helvetica", 3.5)

    def test_long_details_cannot_shrink_below_minimum(self):
        pdf = self.draw_details(5, "ChArUco board details " * 100)
        pdf.setFont.assert_called_once_with("Helvetica", 3.5)

    def test_larger_margin_keeps_larger_font(self):
        pdf = self.draw_details(5, "ChArUco 5x5 | square 30mm")
        pdf.setFont.assert_called_once_with("Helvetica", _mm_to_points(5) * 0.5)


class TiledPaperTests(unittest.TestCase):
    def generate(self, *args: str):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "board.pdf"
            with (
                patch("camera_calibration.generate_charuco._write_tiled_pdf") as writer,
                patch("camera_calibration.generate_charuco._write_minimap"),
                redirect_stdout(io.StringIO()),
            ):
                main([*args, "--output", str(output)])
            return writer.call_args, json.loads(output.with_suffix(".board.json").read_text())

    def test_same_paper_with_margin_uses_one_page_and_keeps_exact_squares(self):
        call, board = self.generate(
            "--squares-x", "5", "--squares-y", "5", "--paper", "A3",
            "--crop-mark", "5", "--square-size", "30", "--tile-paper", "A3",
            "--margin", "2", "--dpi", "400",
        )
        self.assertEqual((call.kwargs["cols"], call.kwargs["rows"]), (1, 1))
        self.assertEqual((call.kwargs["tile_width_mm"], call.kwargs["tile_height_mm"]), (297, 420))
        self.assertEqual(board["square_mm"], 30)
        self.assertEqual(board["marker_mm"], 21)

    def test_same_paper_fill_respects_margin(self):
        call, board = self.generate(
            "--squares-x", "5", "--squares-y", "5", "--paper", "A3",
            "--tile-paper", "A3", "--margin", "2",
        )
        self.assertEqual((call.kwargs["cols"], call.kwargs["rows"]), (1, 1))
        self.assertAlmostEqual(board["square_mm"], 293 / 5)

    def test_custom_bounds_keep_square_size_and_tile_only_checkerboard(self):
        call, board = self.generate(
            "--squares-x", "18", "--squares-y", "25", "--size", "568x785",
            "--square-size", "31", "--tile-paper", "A3", "--margin", "2",
            "--dpi", "400",
        )
        self.assertEqual((call.kwargs["cols"], call.kwargs["rows"]), (2, 2))
        self.assertEqual(call.args[1].shape, (25 * 488, 18 * 488))
        self.assertEqual(board["square_mm"], 31)

    def test_a1_boards_use_four_a3_pages_without_blank_tiles(self):
        for squares_x, squares_y in ((18, 25), (18, 24), (16, 16), (18, 26)):
            with self.subTest(squares_x=squares_x, squares_y=squares_y):
                call, board = self.generate(
                    "--squares-x", str(squares_x), "--squares-y", str(squares_y),
                    "--paper", "A1", "--crop-mark", "2", "--square-size", "30",
                    "--tile-paper", "A3", "--margin", "7", "--dpi", "400",
                )
                self.assertEqual((call.kwargs["cols"], call.kwargs["rows"]), (2, 2))
                self.assertEqual((call.kwargs["tile_width_mm"], call.kwargs["tile_height_mm"]), (297, 420))
                self.assertEqual(board["square_mm"], 30)
                self.assertEqual(call.kwargs["col_spans_px"], [9 * 472, (squares_x - 9) * 472])
                self.assertEqual(call.kwargs["row_spans_px"], [13 * 472, (squares_y - 13) * 472])
                image = call.args[1]
                y = 0
                for height in call.kwargs["row_spans_px"]:
                    x = 0
                    for width in call.kwargs["col_spans_px"]:
                        self.assertEqual(image[y:y + height, x:x + width].min(), 0)
                        x += width
                    y += height
                self.assertEqual(y, image.shape[0])
                self.assertEqual(x, image.shape[1])

    def test_exceeding_four_sheet_capacity_requires_more_pages(self):
        for squares_x, squares_y in ((19, 26), (18, 27)):
            with self.subTest(squares_x=squares_x, squares_y=squares_y):
                call, _ = self.generate(
                    "--squares-x", str(squares_x), "--squares-y", str(squares_y),
                    "--paper", "A1", "--square-size", "30", "--tile-paper", "A3",
                    "--margin", "7", "--dpi", "400",
                )
                self.assertGreater(call.kwargs["cols"] * call.kwargs["rows"], 4)

    def test_exact_squares_must_fit_inside_named_paper_margin(self):
        with self.assertRaisesRegex(SystemExit, "does not fit"):
            self.generate(
                "--squares-x", "5", "--squares-y", "5", "--paper", "A3",
                "--square-size", "59", "--tile-paper", "A3", "--margin", "2",
            )

    def test_fit_error_explains_requested_board_and_alternatives(self):
        with self.assertRaises(SystemExit) as error:
            self.generate(
                "--squares-x", "18", "--squares-y", "25", "--paper", "A3",
                "--crop-mark", "2", "--square-size", "30", "--tile-paper", "A3",
                "--margin", "7", "--dpi", "400",
            )
        message = str(error.exception)
        self.assertIn("18 x 25 squares at 30mm = 540 x 750mm", message)
        self.assertIn("Available: 283 x 406mm (--paper A3, 7mm margin on each edge)", message)
        self.assertIn("--square-size 15.722222 or smaller", message)
        self.assertIn("at most 9 x 13 squares fit", message)
        self.assertIn("replace --paper with --size 540x750", message)

    def test_custom_size_fit_error_uses_exact_canvas_without_subtracting_margins(self):
        with self.assertRaises(SystemExit) as error:
            self.generate(
                "--squares-x", "18", "--squares-y", "25", "--size", "500x700",
                "--square-size", "30", "--tile-paper", "A3", "--margin", "7",
            )
        message = str(error.exception)
        self.assertIn("Available: 500 x 700mm (--size 500x700)", message)
        self.assertIn("--square-size 27.777777 or smaller", message)
        self.assertIn("at most 16 x 23 squares fit", message)
        self.assertIn("replace --size with --size 540x750", message)


if __name__ == "__main__":
    unittest.main()
