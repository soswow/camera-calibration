"""Calibration result: JSON + ROS/OpenCV camera_info YAML."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

# Full-frame (35mm film) sensor size used for equivalent-focal-length conversion.
FULL_FRAME_WIDTH_MM = 36.0
FULL_FRAME_HEIGHT_MM = 24.0
FULL_FRAME_DIAGONAL_MM = math.hypot(FULL_FRAME_WIDTH_MM, FULL_FRAME_HEIGHT_MM)

BOARD_CHECKERBOARD = "checkerboard"
BOARD_CHARUCO = "charuco"


def _flatten_numeric(value) -> list[float]:
    """Flatten nested lists/arrays of numbers into a 1D float list."""
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    flattened: list[float] = []
    for item in value:
        flattened.extend(_flatten_numeric(item))
    return flattened


@dataclass
class CalibrationResult:
    """Intrinsic camera parameters estimated from a calibration board."""

    image_size: tuple[int, int]  # (width, height)
    camera_matrix: list[list[float]]
    distortion_coefficients: list[float]
    rms_reprojection_error: float
    used_images: list[str]
    failed_images: list[str]
    pattern_size: tuple[int, int]  # square counts (squares_x, squares_y)
    square_size: float
    distortion_model: str = "full"
    board_type: str = BOARD_CHECKERBOARD
    dictionary: str | None = None
    marker_proportion: float | None = None
    rotated_images: list[str] | None = None
    auto_select_rejected: list[str] | None = None
    auto_select_threshold_px: float | None = None
    initial_rms_reprojection_error: float | None = None
    fitzgibbon_lambda: float | None = None
    fitzgibbon_rms_reprojection_error: float | None = None

    @property
    def fx(self) -> float:
        return self.camera_matrix[0][0]

    @property
    def fy(self) -> float:
        return self.camera_matrix[1][1]

    @property
    def cx(self) -> float:
        return self.camera_matrix[0][2]

    @property
    def cy(self) -> float:
        return self.camera_matrix[1][2]

    @property
    def hfov_deg(self) -> float:
        """Horizontal field of view in degrees (pinhole model from K)."""
        width, _ = self.image_size
        return math.degrees(2.0 * math.atan(width / (2.0 * self.fx)))

    @property
    def vfov_deg(self) -> float:
        """Vertical field of view in degrees (pinhole model from K)."""
        _, height = self.image_size
        return math.degrees(2.0 * math.atan(height / (2.0 * self.fy)))

    @property
    def dfov_deg(self) -> float:
        """Diagonal field of view in degrees (pinhole model from K)."""
        width, height = self.image_size
        half_diag = math.hypot(width / (2.0 * self.fx), height / (2.0 * self.fy))
        return math.degrees(2.0 * math.atan(half_diag))

    @property
    def focal_length_35mm_equiv(self) -> float:
        """
        35mm-equivalent focal length (mm).

        Matches the calibrated diagonal FOV on a 36×24 mm full-frame sensor.
        """
        return (FULL_FRAME_DIAGONAL_MM / 2.0) / math.tan(
            math.radians(self.dfov_deg) / 2.0
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.update(
            {
                "hfov_deg": self.hfov_deg,
                "vfov_deg": self.vfov_deg,
                "dfov_deg": self.dfov_deg,
                "focal_length_35mm_equiv": self.focal_length_35mm_equiv,
            }
        )
        return payload

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def to_ros_yaml(self, camera_name: str = "camera") -> str:
        """
        OpenCV/ROS camera_info YAML (plumb_bob / Brown-Conrady).

        Compatible with ROS camera_calibration_parsers and many tools that
        expect ~/.ros/camera_info/<name>.yaml style files.
        """
        width, height = self.image_size
        dist = list(self.distortion_coefficients[:5])
        while len(dist) < 5:
            dist.append(0.0)

        def matrix_block(name: str, rows: int, cols: int, data: list[float]) -> str:
            values = ", ".join(repr(float(value)) for value in data)
            return (
                f"{name}:\n"
                f"  rows: {rows}\n"
                f"  cols: {cols}\n"
                f"  data: [{values}]\n"
            )

        k_data = [
            self.fx, 0.0, self.cx,
            0.0, self.fy, self.cy,
            0.0, 0.0, 1.0,
        ]
        r_data = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        p_data = [
            self.fx, 0.0, self.cx, 0.0,
            0.0, self.fy, self.cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

        body = (
            f"image_width: {width}\n"
            f"image_height: {height}\n"
            f"camera_name: {camera_name}\n"
            f"{matrix_block('camera_matrix', 3, 3, k_data)}"
            f"distortion_model: plumb_bob\n"
            f"{matrix_block('distortion_coefficients', 1, 5, dist)}"
            f"{matrix_block('rectification_matrix', 3, 3, r_data)}"
            f"{matrix_block('projection_matrix', 3, 4, p_data)}"
        )
        if self.fitzgibbon_lambda is not None:
            body += f"fitzgibbon_lambda: {repr(float(self.fitzgibbon_lambda))}\n"
        return body

    def save_ros_yaml(self, path: Path, camera_name: str = "camera") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_ros_yaml(camera_name=camera_name))

    @classmethod
    def from_dict(cls, data: dict) -> CalibrationResult:
        """Rebuild from project JSON, including live ChArUco session dumps."""
        camera_matrix = data.get("camera_matrix")
        if camera_matrix is None:
            raise ValueError("Calibration JSON is missing camera_matrix")

        dist = data.get("distortion_coefficients")
        if dist is None:
            dist = data.get("dist_coeffs")
        if dist is None:
            raise ValueError("Calibration JSON is missing distortion coefficients")
        distortion_coefficients = _flatten_numeric(dist)

        rms = data.get("rms_reprojection_error")
        if rms is None:
            rms = data.get("reprojection_error", 0.0)

        board = data.get("board") if isinstance(data.get("board"), dict) else {}
        board_type = str(data.get("board_type") or board.get("type") or BOARD_CHECKERBOARD)
        if "squares_x" in board and "board_type" not in data:
            board_type = BOARD_CHARUCO

        if "pattern_size" in data:
            pattern_size = (int(data["pattern_size"][0]), int(data["pattern_size"][1]))
        elif board:
            pattern_size = (int(board["squares_x"]), int(board["squares_y"]))
        else:
            pattern_size = (0, 0)

        square_size = data.get("square_size")
        if square_size is None:
            square_size = board.get("square_size_mm", 0.0)

        dictionary = data.get("dictionary")
        if dictionary is None:
            dictionary = board.get("dictionary")

        marker_proportion = data.get("marker_proportion")
        if marker_proportion is None and board.get("square_size_mm") and board.get("marker_size_mm"):
            marker_proportion = float(board["marker_size_mm"]) / float(board["square_size_mm"])

        used_images = data.get("used_images")
        if used_images is None and data.get("frames_used") is not None:
            used_images = [str(index) for index in range(int(data["frames_used"]))]

        return cls(
            image_size=(int(data["image_size"][0]), int(data["image_size"][1])),
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients,
            rms_reprojection_error=float(rms),
            used_images=list(used_images or []),
            failed_images=list(data.get("failed_images", [])),
            pattern_size=pattern_size,
            square_size=float(square_size or 0.0),
            distortion_model=str(data.get("distortion_model", "full")),
            board_type=board_type,
            dictionary=str(dictionary) if dictionary is not None else None,
            marker_proportion=(
                float(marker_proportion) if marker_proportion is not None else None
            ),
            rotated_images=list(data["rotated_images"]) if data.get("rotated_images") else [],
            auto_select_rejected=(
                list(data["auto_select_rejected"]) if data.get("auto_select_rejected") else None
            ),
            auto_select_threshold_px=(
                float(data["auto_select_threshold_px"])
                if data.get("auto_select_threshold_px") is not None
                else None
            ),
            initial_rms_reprojection_error=(
                float(data["initial_rms_reprojection_error"])
                if data.get("initial_rms_reprojection_error") is not None
                else None
            ),
            fitzgibbon_lambda=(
                float(data["fitzgibbon_lambda"])
                if data.get("fitzgibbon_lambda") is not None
                else None
            ),
            fitzgibbon_rms_reprojection_error=(
                float(data["fitzgibbon_rms_reprojection_error"])
                if data.get("fitzgibbon_rms_reprojection_error") is not None
                else None
            ),
        )

    @classmethod
    def from_json(cls, path: Path) -> CalibrationResult:
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def from_yaml(cls, path: Path) -> CalibrationResult:
        """Load ROS/OpenCV camera_info YAML written by save_ros_yaml."""
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError(
                "PyYAML is required to load calibration YAML. "
                "Install with: pip install pyyaml"
            ) from error

        try:
            payload = yaml.safe_load(path.read_text())
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML in {path}: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a YAML mapping in {path}")

        try:
            width = int(payload["image_width"])
            height = int(payload["image_height"])
            k_data = list(payload["camera_matrix"]["data"])
            d_data = [float(value) for value in payload["distortion_coefficients"]["data"]]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"YAML {path} is missing camera_info fields "
                "(image_width/height, camera_matrix.data, distortion_coefficients.data)"
            ) from error

        if len(k_data) != 9:
            raise ValueError(f"camera_matrix.data must have 9 values, got {len(k_data)}")

        camera_matrix = [
            [float(k_data[0]), float(k_data[1]), float(k_data[2])],
            [float(k_data[3]), float(k_data[4]), float(k_data[5])],
            [float(k_data[6]), float(k_data[7]), float(k_data[8])],
        ]
        fitzgibbon_lambda = payload.get("fitzgibbon_lambda")
        yaml_model = str(payload.get("distortion_model", "plumb_bob"))

        return cls(
            image_size=(width, height),
            camera_matrix=camera_matrix,
            distortion_coefficients=d_data,
            rms_reprojection_error=0.0,
            used_images=[],
            failed_images=[],
            pattern_size=(0, 0),
            square_size=0.0,
            distortion_model=yaml_model,
            fitzgibbon_lambda=(
                float(fitzgibbon_lambda) if fitzgibbon_lambda is not None else None
            ),
        )

    @classmethod
    def from_path(cls, path: Path) -> CalibrationResult:
        """Load JSON (project format) or ROS camera_info YAML by file suffix."""
        suffix = path.suffix.lower()
        if suffix == ".json":
            return cls.from_json(path)
        if suffix in {".yaml", ".yml"}:
            return cls.from_yaml(path)
        raise ValueError(
            f"Unsupported calibration file {path} "
            "(expected .json, .yaml, or .yml)"
        )
