from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from camera_calibration.board import make_board
from camera_calibration.charuco import find_charuco_corners
from camera_calibration.detection import (
    DetectedView,
    DetectionSet,
    fit_intrinsics,
    object_points_grid,
)
from camera_calibration.images import (
    read_calibration_image,
    normalize_to_calibration_size,
)
from camera_calibration.models import CameraModel, model_for
from camera_calibration.observations import (
    partition,
    save_observations,
    load_observations,
)
from camera_calibration.render import remap_view
from camera_calibration.result import CalibrationResult
from camera_calibration.validate import validate_view, render_validation_overlay


def result(asymmetric=False):
    return CalibrationResult(
        (800, 600),
        [[220.0, 0, 399.5], [0, 225.0, 299.5], [0, 0, 1]],
        [0.04, -0.003],
        0.0,
        [],
        [],
        (7, 5),
        30.0,
        distortion_model="angular-asymmetric" if asymmetric else "angular",
        asymmetric_coefficients=(0.001, -0.0007) if asymmetric else (0.0, 0.0),
    )


def synthetic_views(count=18, asymmetric=False):
    # Independent formula, rectangular raster, fixed RNG and diverse target poses.
    rng = np.random.default_rng(19283)
    objects = object_points_grid(6, 4, 30).astype(float)
    center = objects.mean(axis=0)
    views = []
    for index in range(count):
        rv = rng.uniform(-0.65, 0.65, 3)
        rotation = cv2.Rodrigues(rv)[0]
        target = np.r_[rng.uniform(-200, 200, 2), rng.uniform(200, 430)]
        tv = target - rotation @ center
        xyz = objects @ rotation.T + tv
        r = np.hypot(xyz[:, 0], xyz[:, 1])
        t = np.arctan2(r, xyz[:, 2])
        rho = t * (1 + 0.04 * t * t - 0.003 * t**4)
        x, y = (xyz[:, :2] * (rho / r)[:, None]).T
        if asymmetric:
            p1, p2 = 0.001, -0.0007
            xd = x + 2 * p1 * x * y + p2 * (3 * x * x + y * y)
            yd = y + p1 * (x * x + 3 * y * y) + 2 * p2 * x * y
        else:
            xd, yd = x, y
        pixels = np.column_stack((220 * xd + 399.5, 225 * yd + 299.5))
        views.append(DetectedView(f"{index}.png", objects, pixels[:, None, :], False))
    return DetectionSet((800, 600), (7, 5), 30.0, views, [])


class ProjectionTests(unittest.TestCase):
    def test_rays_beyond_90_and_asymmetric_roundtrip(self):
        for asymmetric in (False, True):
            model = model_for(result(asymmetric))
            theta = np.deg2rad([0, 45, 85, 95, 105, 109.9])
            rays = np.column_stack((np.sin(theta), np.zeros(len(theta)), np.cos(theta)))
            pixels = model.project_rays(rays)
            np.testing.assert_allclose(model.unproject_pixels(pixels), rays, atol=1e-10)
            self.assertTrue(np.all(np.diff(pixels[:, 0]) > 0))

    def test_reject_unknown_nonmonotonic_and_wrong_export(self):
        data = result().to_dict()
        for change in [
            {"distortion_model": "equidistant"},
            {"distortion_coefficients": [-1.0, 0.0]},
            {"schema_version": 99},
        ]:
            with self.assertRaises(ValueError):
                CalibrationResult.from_dict(data | change)
        with self.assertRaises(ValueError):
            result().to_ros_yaml()
        self.assertIsNone(result().hfov_deg)
        np.testing.assert_allclose(
            model_for(CalibrationResult.from_dict(data)).coefficients, [0.04, -0.003]
        )
        with self.assertRaises(ValueError):
            model_for(result()).unproject_pixels([[1e6, 1e6]])

    def test_brown_matches_opencv(self):
        m = CameraModel(
            [[500, 0, 320], [0, 510, 240], [0, 0, 1]], [-0.1, 0.02, 0.001, -0.002, 0]
        )
        rays = np.array([[0.1, 0.2, 1], [-0.3, 0.1, 1], [0, 0, 1.0]])
        expected = cv2.projectPoints(
            rays, np.zeros(3), np.zeros(3), m.matrix, m.coefficients
        )[0]
        np.testing.assert_allclose(m.project_rays(rays), expected.reshape(-1, 2))

    def test_remap_direction_center_and_domain(self):
        c = result()
        image = np.zeros((600, 800, 3), np.uint8)
        image[:, :, 0] = np.arange(800, dtype=np.uint16)[None, :] % 256
        mapped, valid = remap_view(image, c, output_size=(101, 101))
        self.assertTrue(valid[50, 50])
        self.assertLess(abs(int(mapped[50, 50, 0]) - 144), 2)
        _, valid = remap_view(
            image, c, projection="equirectangular", output_size=(200, 100)
        )
        self.assertFalse(valid[50, 0])
        self.assertTrue(valid[50, 100])
        with self.assertRaises(ValueError):
            remap_view(image, c, fov=180)

    def test_encoded_and_legacy_pixels(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "orientation.png"
            image = np.arange(24, dtype=np.uint8).reshape(4, 6)
            exif = Image.Exif()
            exif[274] = 6
            Image.fromarray(image).save(path, exif=exif)
            self.assertEqual(read_calibration_image(path).image.shape[:2], (4, 6))
            self.assertEqual(
                read_calibration_image(path, "legacy-exif").image.shape[:2], (6, 4)
            )
            self.assertIsNone(
                normalize_to_calibration_size(np.zeros((4, 6, 3)), (4, 6))
            )
            old = result().to_dict()
            old.pop("pixel_policy")
            self.assertEqual(
                CalibrationResult.from_dict(old).pixel_policy, "legacy-exif"
            )


class CalibrationTests(unittest.TestCase):
    def test_radial_rectangular_recovery_and_validation(self):
        detections = synthetic_views()
        fit = fit_intrinsics(
            DetectionSet(**{**detections.__dict__, "views": detections.views[:15]}),
            "angular",
        )
        np.testing.assert_allclose(
            [fit.fx, fit.fy, fit.cx, fit.cy], [220, 225, 399.5, 299.5], atol=0.03
        )
        for v in detections.views[15:]:
            metrics, _, _ = validate_view(v, fit)
            self.assertLess(metrics.reprojection.rms_px, 0.002)
            self.assertLess(metrics.spherical_straightness["rms_deg"], 0.001)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "overlay.png"
            v = detections.views[-1]
            r, _, _ = validate_view(v, fit)
            render_validation_overlay(
                np.zeros((600, 800, 3), np.uint8), v, fit, r, path
            )
            self.assertIsNotNone(cv2.imread(str(path)))

    def test_asymmetric_recovery(self):
        fit = fit_intrinsics(synthetic_views(24, True), "angular-asymmetric")
        np.testing.assert_allclose(
            fit.asymmetric_coefficients, [0.001, -0.0007], atol=2e-5
        )
        self.assertLess(fit.rms_reprojection_error, 0.005)

    def test_cache_and_duplicate_partition(self):
        data = synthetic_views(12)
        duplicate = DetectedView(
            "copy.png",
            data.views[0].object_points,
            data.views[0].image_points.copy(),
            False,
        )
        data.views.append(duplicate)
        train, valid, excluded = partition(data, reserved=["copy.png", "5.png"])
        self.assertNotIn("0.png", [v.name for v in train.views])
        self.assertIn("0.png", excluded)
        self.assertEqual({v.name for v in valid}, {"copy.png", "5.png"})
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "observations.json"
            save_observations(data, p)
            loaded = load_observations(p)
            np.testing.assert_array_equal(
                loaded.views[0].image_points, data.views[0].image_points
            )
        with self.assertRaises(ValueError):
            partition(data, reserved=["missing.png"])

    def test_multiscale_actual_board_and_shared_ids(self):
        board = make_board(
            7,
            5,
            30,
            21,
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100),
            first_marker_id=10,
        )
        image = board.generateImage((1400, 1000), marginSize=40)
        original = image.copy()
        detector = cv2.aruco.CharucoDetector(board)

        class FailNative:
            def detectBoard(self, gray):
                if gray.shape == image.shape:
                    return None, None, (), None
                return detector.detectBoard(gray)

        with patch(
            "camera_calibration.charuco.aruco.CharucoDetector",
            return_value=FailNative(),
        ):
            found = find_charuco_corners(image, board)
        self.assertIsNotNone(found)
        self.assertEqual(len(found[1]), 24)
        np.testing.assert_array_equal(image, original)
        self.assertEqual(int(board.getIds()[0]), 10)


class WorkflowTests(unittest.TestCase):
    def test_masked_detection_on_rectangular_image(self):
        from camera_calibration.charuco import collect_detections

        board = make_board(
            7, 5, 30, 21, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        )
        patch_image = board.generateImage((560, 400), marginSize=20)
        image = np.full((500, 1300), 255, np.uint8)
        image[50:450, 30:590] = patch_image
        image[50:450, 710:1270] = patch_image
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "two.png"
            Image.fromarray(image).save(p)
            data = collect_detections(
                Path(tmp),
                7,
                5,
                30,
                0.7,
                "DICT_4X4_100",
                min_views=1,
                masks={"two.png": [[[0.52, 0], [1, 0], [1, 1], [0.52, 1]]]},
            )
            self.assertEqual(data.image_size, (1300, 500))
            self.assertEqual(len(data.views), 1)
            self.assertLess(float(data.views[0].image_points[:, :, 0].max()), 600)
            self.assertTrue(data.views[0].source_sha256)
            np.testing.assert_array_equal(np.asarray(Image.open(p)), image)

    def test_rim_board_pose(self):
        model = model_for(result(True))
        objects = object_points_grid(6, 4, 30).astype(float)
        rotation = cv2.Rodrigues(np.array([0.0, np.deg2rad(95), 0.0]))[0]
        target = rotation @ (np.array([0.0, 0.0, 1000.0]) - objects.mean(axis=0))
        pose = np.r_[cv2.Rodrigues(rotation)[0].ravel(), target]
        pixels = model.project(objects, pose)
        fitted = model.estimate_board_pose(objects, pixels)
        np.testing.assert_allclose(model.project(objects, fitted), pixels, atol=1e-5)

    def test_brown_cli_cache_export_and_render(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from camera_calibration.cli.main import main
        from camera_calibration.undistort import undistort_image

        data = synthetic_views(15)
        rng = np.random.default_rng(42)
        matrix = np.array([[500.0, 0, 400], [0, 510, 300], [0, 0, 1]])
        for view in data.views:
            rv = rng.uniform(-0.5, 0.5, 3)
            tv = np.r_[rng.uniform(-120, 0, 2), rng.uniform(400, 600)]
            view.object_points = view.object_points.astype(np.float32)
            view.image_points = cv2.projectPoints(
                view.object_points, rv, tv, matrix, np.array([-0.1, 0.02, 0, 0, 0.0])
            )[0]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_observations(data, root / "cache.json")
            with redirect_stdout(StringIO()):
                status = main(
                    [
                        "calibrate",
                        str(root),
                        "--observations-in",
                        str(root / "cache.json"),
                        "--model",
                        "simple",
                        "--output-folder",
                        str(root),
                        "--output-name",
                        "brown",
                    ]
                )
            self.assertEqual(status, 0)
            calibration = CalibrationResult.from_path(root / "brown.json")
            self.assertEqual(calibration.pixel_policy, "encoded")
            self.assertEqual(
                CalibrationResult.from_path(root / "brown.yaml").pixel_policy, "encoded"
            )
            self.assertLess(calibration.validation["rms_px"], 0.001)
            self.assertIn(
                "distortion_model: plumb_bob", (root / "brown.yaml").read_text()
            )
            self.assertEqual(
                undistort_image(np.zeros((600, 800, 3), np.uint8), calibration).shape,
                (600, 800, 3),
            )

    def test_generator_definition_and_legacy_pattern(self):
        from camera_calibration.cli.main import main
        from contextlib import redirect_stdout
        from io import StringIO

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with redirect_stdout(StringIO()):
                status = main(
                    [
                        "generate-charuco",
                        "--squares-x",
                        "7",
                        "--squares-y",
                        "5",
                        "--square-size",
                        "30",
                        "--dictionary",
                        "DICT_4X4_100",
                        "--first-marker-id",
                        "10",
                        "--legacy-pattern",
                        "--dpi",
                        "60",
                        "--output",
                        str(root / "board.png"),
                    ]
                )
            self.assertEqual(status, 0)
            import json

            definition = json.loads((root / "board.board.json").read_text())
            self.assertTrue(definition["legacy_pattern"])
            self.assertEqual(definition["first_marker_id"], 10)
            board = make_board(
                7,
                5,
                30,
                21,
                cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100),
                legacy_pattern=True,
                first_marker_id=10,
            )
            corners = find_charuco_corners(
                cv2.imread(str(root / "board.png"), cv2.IMREAD_GRAYSCALE), board
            )
            self.assertIsNotNone(corners)

    def test_auto_select_uses_angular_model(self):
        from camera_calibration.auto_select import score_views

        data = synthetic_views(12, True)
        model = model_for(result(True))
        scores = score_views(data, model.matrix, model.coefficients, model)
        self.assertLess(max(s.mean_error_px for s in scores), 1e-5)

    def test_invalid_angular_asymmetry_domain(self):
        data = result(True).to_dict()
        data["asymmetric_coefficients"] = [1.0, 0.0]
        with self.assertRaises(ValueError):
            CalibrationResult.from_dict(data)


class RejectionTests(unittest.TestCase):
    def test_repeated_pose_does_not_establish_intrinsics(self):
        data = synthetic_views(12)
        for view in data.views:
            view.image_points = data.views[0].image_points.copy()
        with self.assertRaises(ValueError):
            fit_intrinsics(data, "angular")

    def test_excluded_duplicate_is_not_held_out(self):
        data = synthetic_views(1)
        calibration = result()
        calibration.validation = {"excluded_duplicates": [data.views[0].name]}
        record, _, _ = validate_view(data.views[0], calibration)
        self.assertTrue(record.used_for_calibration)


if __name__ == "__main__":
    unittest.main()
