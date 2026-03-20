"""Camera thread - captures RGB images from a USB webcam (OpenCV)."""

import time
import traceback

import cv2
from common.configs import (
    CAMERA_DEVICE_INDEX,
    CAMERA_FRAME_STREAMING_RATE,
    CAMERA_HEIGHT,
    CAMERA_NAMES,
    CAMERA_WIDTH,
)
from common.data_manager import DataManager


def camera_thread(data_manager: DataManager) -> None:
    """Camera thread - captures RGB images from a USB webcam."""
    print("📷 Camera thread started (USB webcam)")

    dt: float = 1.0 / CAMERA_FRAME_STREAMING_RATE
    cap: cv2.VideoCapture | None = None
    camera_name = CAMERA_NAMES[1]

    try:
        cap = cv2.VideoCapture(CAMERA_DEVICE_INDEX)
        if not cap.isOpened():
            print(
                f"❌ Could not open USB webcam (device index {CAMERA_DEVICE_INDEX}). "
                "Check connection and permissions."
            )
            data_manager.request_shutdown()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FRAME_STREAMING_RATE)

        # Read back actual resolution (some webcams don't support requested size)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"  Webcam opened: {actual_w}x{actual_h} @ ~{CAMERA_FRAME_STREAMING_RATE} Hz"
        )

        while not data_manager.is_shutdown_requested():
            iteration_start = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️  Webcam read failed, skipping frame")
                time.sleep(dt)
                continue

            # OpenCV returns BGR; convert to RGB for consistency with pipeline
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Camera may be mounted upside down; rotate 180° to keep images upright
            rgb_image = cv2.rotate(rgb_image, cv2.ROTATE_180)
            # Treat USB webcam as wrist camera
            data_manager.set_rgb_image(rgb_image, camera_name)

            elapsed = time.time() - iteration_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"❌ Camera thread error: {e}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        if cap is not None:
            cap.release()
            print("  ✓ USB webcam released")
        print("📷 Camera thread stopped")
