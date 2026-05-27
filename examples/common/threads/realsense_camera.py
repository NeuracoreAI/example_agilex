"""Camera thread - captures RGB images from RealSense with drop detection."""

import time
import traceback
from collections import deque

import numpy as np
import pyrealsense2 as rs
from common.configs import CAMERA_FRAME_STREAMING_RATE, CAMERA_NAMES
from common.data_manager import DataManager


def camera_thread(data_manager: DataManager) -> None:
    """Camera thread - captures RGB images from RealSense and monitors health."""
    print("📷 Camera thread started")

    camera_name = CAMERA_NAMES[0]
    pipeline: rs.pipeline | None = None

    # Diagnostic variables
    last_frame_number = None
    total_dropped_frames = 0
    fps_timer = time.time()
    frame_count = 0
    # Store the last 100 frame intervals to check for extreme jitter
    intervals = deque(maxlen=100)
    last_frame_time = time.time()

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
            try:
                # wait_for_frames naturally blocks at the target framerate (e.g., 60Hz)
                frames = pipeline.wait_for_frames(timeout_ms=500)
            except Exception as e:
                print(f"⚠️  RealSense wait for frames error (Timeout?): {e}")
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            current_time = time.time()
            intervals.append(current_time - last_frame_time)
            last_frame_time = current_time

            # ---------------------------------------------------------
            # DIAGNOSTICS: Check for dropped frames via hardware ID
            # ---------------------------------------------------------
            current_frame_number = color_frame.get_frame_number()

            if last_frame_number is not None:
                # If frame numbers are not sequential, we missed something in software
                drops = (current_frame_number - last_frame_number) - 1
                if drops > 0:
                    total_dropped_frames += drops
                    print(
                        f"⚠️  DROPPED {drops} FRAME(S)! (Hardware ID: {current_frame_number}) | Total dropped: {total_dropped_frames}"
                    )

            last_frame_number = current_frame_number

            # ---------------------------------------------------------
            # DIAGNOSTICS: Calculate software-side FPS and Jitter
            # ---------------------------------------------------------
            frame_count += 1
            if current_time - fps_timer >= 5.0:  # Report every 5 seconds
                effective_fps = frame_count / (current_time - fps_timer)

                # Calculate jitter (max variance between frame arrivals)
                max_interval = max(intervals) * 1000  # in ms
                min_interval = min(intervals) * 1000  # in ms

                print(
                    f"📊 Camera Health: {effective_fps:.1f} FPS | "
                    f"Jitter: {min_interval:.1f}ms - {max_interval:.1f}ms | "
                    f"Total Drops: {total_dropped_frames}"
                )

                # Reset counters for the next 5-second window
                fps_timer = current_time
                frame_count = 0

            # ---------------------------------------------------------
            # IMAGE PROCESSING
            # ---------------------------------------------------------
            # color_image = np.asanyarray(color_frame.get_data())
            # color_image = np.rot90(color_image, k=3)  # Rotate 270 degrees
            # data_manager.set_rgb_image(color_image, camera_name)

            color_image = np.asanyarray(color_frame.get_data())
            color_image = np.rot90(color_image, k=3)  # Rotate 270 degrees
            data_manager.set_rgb_image(color_image, camera_name)

            # Notice: The time.sleep() has been completely removed.
            # wait_for_frames() manages the loop pace perfectly.

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
