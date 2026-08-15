#!/usr/bin/env python3
"""Simple PS3 Eye preview (no CLI options)."""

from __future__ import annotations

# ---- Configuration ----
DEVICE_INDEX = 0
WIDTH = 640
HEIGHT = 480
FPS = 15
FORMAT = "raw"  # "bgr" or "raw"
QUEUE_SIZE = 2
TIMEOUT_MS = 1000
PREVIEW_SCALE = 2
RETINA_SCALE = 2
# Exposure settings
AUTO_EXPOSURE = False
GAIN = 26
EXPOSURE = 154
# -----------------------


def main() -> int:
    try:
        import ps3eye  # type: ignore
    except Exception as exc:
        print("Failed to import ps3eye. Ensure the module is on PYTHONPATH.")
        print(f"Error: {exc}")
        return 1

    try:
        import cv2
    except Exception as exc:
        print("OpenCV is required for preview.")
        print(f"Error: {exc}")
        return 1

    devices = ps3eye.list_devices()
    if not devices:
        print("No PS3 Eye devices found.")
        return 1

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

        cv2.namedWindow("ps3eye", cv2.WINDOW_NORMAL)
        preview_ready = False
        effective_scale = PREVIEW_SCALE * RETINA_SCALE
        while True:
            frame = dev.read(timeout_ms=TIMEOUT_MS)
            if frame is None:
                continue
            img = frame.to_numpy(copy=False)
            preview = cv2.resize(
                img,
                (img.shape[1] * effective_scale, img.shape[0] * effective_scale),
                interpolation=cv2.INTER_NEAREST,
            )
            if not preview_ready:
                cv2.resizeWindow("ps3eye", preview.shape[1], preview.shape[0])
                preview_ready = True
            cv2.imshow("ps3eye", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        dev.stop_stream()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
