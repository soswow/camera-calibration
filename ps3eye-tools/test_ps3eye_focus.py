#!/usr/bin/env python3
"""Live focus preview with checkerboard detection."""

from __future__ import annotations

from collections import deque
import time

# ---- Configuration ----
DEVICE_INDEX = 0
WIDTH = 640
HEIGHT = 480
FPS = 30
FORMAT = "raw"  # "bgr" or "raw"
QUEUE_SIZE = 2
TIMEOUT_MS = 1000
PREVIEW_SCALE = 2
RETINA_SCALE = 1
AUTO_EXPOSURE = False
GAIN = 40
EXPOSURE = 60

# Checkerboard settings (count squares; OpenCV uses inner corners = squares - 1)
CHECKERBOARD_SQUARES_X = 7
CHECKERBOARD_SQUARES_Y = 10

# Rolling average
SHARPNESS_WINDOW = 120
LOCK_SECONDS = 5.0
# -----------------------

def _sharpness(gray, bbox) -> float:
    import cv2

    if bbox:
        x0, y0, x1, y1 = bbox
        roi = gray[y0 : y1 + 1, x0 : x1 + 1]
    else:
        roi = gray
    if roi.size == 0:
        return 0.0
    lap = cv2.Laplacian(roi, cv2.CV_64F)
    return float(lap.var())


def main() -> int:
    try:
        import ps3eye  # type: ignore
    except Exception as exc:
        print("Failed to import ps3eye. Ensure the module is on PYTHONPATH.")
        print(f"Error: {exc}")
        return 1

    try:
        import cv2
        import numpy as np
    except Exception as exc:
        print("OpenCV is required for preview.")
        print(f"Error: {exc}")
        return 1

    devices = ps3eye.list_devices()
    if not devices:
        print("No PS3 Eye devices found.")
        return 1

    effective_scale = PREVIEW_SCALE * RETINA_SCALE
    preview_ready = False
    sharp_history: deque[float | None] = deque(maxlen=SHARPNESS_WINDOW)
    lock_start = time.time()
    locked = False
    samples: list[np.ndarray] = []
    avg_corners = None
    avg_bbox = None

    with ps3eye.Device(DEVICE_INDEX) as dev:
        dev.configure(
            width=WIDTH,
            height=HEIGHT,
            fps=FPS,
            format=FORMAT,
        )
        if AUTO_EXPOSURE:
            dev.set_auto(True)
        else:
            dev.set_auto(False)
            dev.set_manual(GAIN, EXPOSURE)
        dev.start_stream(queue_size=QUEUE_SIZE)

        cv2.namedWindow("ps3eye_focus", cv2.WINDOW_NORMAL)
        while True:
            frame = dev.read(timeout_ms=TIMEOUT_MS)
            if frame is None:
                continue

            img = frame.to_numpy(copy=False)
            if img.ndim == 2:
                gray = img
                color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                color = img
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            pattern_size = (CHECKERBOARD_SQUARES_X - 1, CHECKERBOARD_SQUARES_Y - 1)
            found = False
            corners = None

            if not locked:
                found, corners = cv2.findChessboardCorners(
                    gray,
                    pattern_size,
                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
                )
                if found:
                    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
                    samples.append(corners.reshape(-1, 2))

                if time.time() - lock_start >= LOCK_SECONDS:
                    locked = True
                    if samples:
                        avg_corners = sum(samples) / len(samples)
                        avg_bbox = cv2.boundingRect(avg_corners.astype(int))
                    else:
                        avg_corners = None
                        avg_bbox = None

            sharp = 0.0
            if locked and avg_bbox is not None and avg_corners is not None:
                x, y, w, h = avg_bbox
                sharp = _sharpness(gray, (x, y, x + w - 1, y + h - 1))
                rect = cv2.minAreaRect(avg_corners.astype(np.float32))
                box = cv2.boxPoints(rect).astype(int)
                cv2.polylines(color, [box], True, (255, 0, 0), 2)
            elif not locked and found and corners is not None:
                cv2.drawChessboardCorners(color, pattern_size, corners, found)
                rect = cv2.minAreaRect(corners)
                box = cv2.boxPoints(rect).astype(int)
                cv2.polylines(color, [box], True, (255, 0, 0), 2)
                x, y, w, h = cv2.boundingRect(corners.astype(int))
                sharp = _sharpness(gray, (x, y, x + w - 1, y + h - 1))

            if locked:
                sharp_history.append(sharp if avg_bbox is not None else None)
            else:
                sharp_history.append(sharp if found else None)

            valid_vals = [v for v in sharp_history if v is not None]
            avg_sharp = sum(valid_vals) / len(valid_vals) if valid_vals else 0.0

            if locked and avg_bbox is None:
                text = "Sharpness avg: N/A (lock failed)"
                color_fg = (0, 0, 255)
            elif not locked:
                text = f"Sharpness avg: {int(round(avg_sharp))}"
                color_fg = (0, 255, 0) if found else (0, 0, 255)
            else:
                text = f"Sharpness avg: {int(round(avg_sharp))}"
                color_fg = (0, 255, 255)

            cv2.putText(color, text, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(color, text, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_fg, 2, cv2.LINE_AA)
            if not locked:
                remaining = max(0.0, LOCK_SECONDS - (time.time() - lock_start))
                status_text = f"LOCKING {remaining:.1f}s"
                y = color.shape[0] - 12
                cv2.putText(color, status_text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(color, status_text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)

            preview = cv2.resize(
                color,
                (color.shape[1] * effective_scale, color.shape[0] * effective_scale),
                interpolation=cv2.INTER_NEAREST,
            )
            if not preview_ready:
                cv2.resizeWindow("ps3eye_focus", preview.shape[1], preview.shape[0])
                preview_ready = True
            cv2.imshow("ps3eye_focus", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        dev.stop_stream()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
