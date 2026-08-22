from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from camera_calibration.detection import DetectedView, object_points_grid
from camera_calibration.result import CalibrationResult
from camera_calibration.validate import (
    render_validation_overlay,
    straightness_errors,
    validate_view,
)


def _calibration() -> CalibrationResult:
    return CalibrationResult(
        image_size=(640, 480),
        camera_matrix=[[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        distortion_coefficients=[0.0, 0.0, 0.0, 0.0, 0.0],
        rms_reprojection_error=0.7,
        used_images=[],
        failed_images=[],
        pattern_size=(7, 6),
        square_size=1.0,
    )


class ValidationTests(unittest.TestCase):
    def test_validate_view_has_zero_error_for_exact_projection(self) -> None:
        calibration = _calibration()
        object_points = object_points_grid(6, 5, 1.0)
        image_points, _ = cv2.projectPoints(
            object_points,
            np.array([0.12, -0.08, 0.03]),
            np.array([[-2.5], [-2.0], [12.0]]),
            np.asarray(calibration.camera_matrix),
            np.zeros(5),
        )
        view = DetectedView("held-out.png", object_points, image_points, False)

        result, residuals, line_errors = validate_view(view, calibration)

        self.assertLess(result.reprojection.rms_px, 1e-4)
        self.assertLess(float(np.max(residuals)), 1e-4)
        self.assertIsNotNone(result.straightness)
        assert result.straightness is not None
        self.assertLess(result.straightness.rms_px, 1e-4)
        self.assertGreater(line_errors.size, 0)

    def test_straightness_reports_curved_grid(self) -> None:
        object_points = object_points_grid(6, 5, 1.0)
        points = object_points[:, :2].astype(np.float64) * 40.0
        points[:, 1] += 0.15 * np.square(points[:, 0] - points[:, 0].mean())

        errors, row_lines, column_lines = straightness_errors(object_points, points)

        self.assertEqual(row_lines, 5)
        self.assertEqual(column_lines, 6)
        self.assertGreater(float(np.sqrt(np.mean(np.square(errors)))), 1.0)

    def test_render_validation_overlay(self) -> None:
        calibration = _calibration()
        object_points = object_points_grid(6, 5, 1.0)
        image_points, _ = cv2.projectPoints(
            object_points,
            np.array([0.12, -0.08, 0.03]),
            np.array([[-2.5], [-2.0], [12.0]]),
            np.asarray(calibration.camera_matrix),
            np.zeros(5),
        )
        view = DetectedView("held-out.png", object_points, image_points, False)
        result, _residuals, _line_errors = validate_view(view, calibration)
        image = np.full((480, 640, 3), 80, dtype=np.uint8)

        with TemporaryDirectory() as folder:
            destination = Path(folder) / "overlay.png"
            render_validation_overlay(image, view, calibration, result, destination)
            rendered = cv2.imread(str(destination))

        self.assertIsNotNone(rendered)
        assert rendered is not None
        self.assertEqual(rendered.shape, image.shape)
        self.assertGreater(int(np.max(rendered)), 80)


if __name__ == "__main__":
    unittest.main()
