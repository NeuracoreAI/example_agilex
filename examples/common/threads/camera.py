"""Camera thread - captures RGB images from RealSense."""

import time
import traceback

import numpy as np
import pyrealsense2 as rs
from common.configs import CAMERA_FRAME_STREAMING_RATE
from common.data_manager import DataManager


def camera_thread(data_manager: DataManager) -> None:
    """Camera thread - captures RGB images from RealSense."""
    print("📷 Camera thread started")

    dt: float = 1.0 / CAMERA_FRAME_STREAMING_RATE
    pipeline: rs.pipeline | None = None

    try:
        # Configure RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(
            rs.stream.color,
            640,
            480,
            rs.format.rgb8,
            int(CAMERA_FRAME_STREAMING_RATE),
        )

        print(f"Starting RealSense pipeline at {CAMERA_FRAME_STREAMING_RATE} Hz...")
        pipeline.start(config)

        while not data_manager.is_shutdown_requested():
            iteration_start = time.time()

            try:
                frames = pipeline.wait_for_frames(timeout_ms=500)
            except Exception as e:
                print(f"⚠️  RealSense wait for frames error: {e}")
                continue

            color_frame = frames.get_color_frame()

            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                color_image = np.rot90(color_image, k=3)  # Rotate 270 degrees
                data_manager.set_rgb_image(color_image)

            # Sleep to maintain loop rate
            elapsed = time.time() - iteration_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"❌ Camera thread error: {e}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
                print("  ✓ RealSense pipeline stopped")
            except Exception as e:
                print(f"⚠️  Error stopping pipeline: {e}")
        print("📷 Camera thread stopped")
