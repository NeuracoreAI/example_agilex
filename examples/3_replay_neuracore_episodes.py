#!/usr/bin/env python3
"""
Piper Robot Dataset Replay and Validation.

This script acts as a production-grade utility for replaying recorded Neuracore
teleoperation datasets directly on the physical AgileX Piper robotic arm.

=============================================================================
⚠️  HALID'S SAFETY NOTES & OPERATIONAL WARNINGS:
    1. DANGER: This step can be dangerous. The robot will start moving on the 
       exact trajectory that was recorded. ALWAYS be ready to press the emergency stop.
    2. EPISODE INDEXING: The episode index starts at 0, not 1. If you see an 
       episode numbered 'X' in the Neuracore frontend, you must run index 'X-1'.
    3. RERUN INSTABILITY: You may experience issues running this command successfully 
       more than once in a row without a system reset. 
    4. FREQUENCY: The default playback frequency should be 20 Hz. Setting the 
       frequency to 0 plays the data aperiodically (as it was recorded).
    5. EMERGENCY STOP: Pressing Ctrl+C will gracefully disable the robot and cut 
       power to the motors after 5 seconds.
=============================================================================

Hardware Requirements:
    - AgileX Piper robot arm connected via CAN interface ('can0').

Usage Examples:
    # Play the first episode (index 0) at the default 20 Hz:
    python 3_replay_neuracore_episodes.py --dataset-name my_dataset
    
    # Play a specific episode index (e.g., frontend episode 3 -> index 2):
    python 3_replay_neuracore_episodes.py --dataset-name my_dataset --episode-index 2
    
    # Play aperiodically (as recorded):
    python 3_replay_neuracore_episodes.py --dataset-name my_dataset --frequency 0
"""

import argparse
import sys
import time
from pathlib import Path
from typing import cast

import cv2
import neuracore as nc
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path Configuration & Local Imports
# ---------------------------------------------------------------------------
# Dynamically append the parent directory to sys.path to resolve local 'common' modules.
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import GRIPPER_NAME, JOINT_NAMES, NEUTRAL_JOINT_ANGLES, ROBOT_RATE
from common.dataset_helpers import load_and_sync_dataset
from neuracore_types import DataType, SynchronizedPoint
from piper_controller import PiperController

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def wait_for_home(robot_controller: PiperController, timeout: int = 10) -> bool:
    """
    Commands the robot to its home position and blocks until it arrives.
    """
    robot_controller.move_to_home()
    for remaining in range(timeout, 0, -1):
        if robot_controller.is_robot_homed():
            print("✓ Robot has safely reached the home position.")
            return True
        print(f"  Waiting for robot to home... ({remaining}s remaining)", end="\r")
        time.sleep(1)
    
    print("\n❌ Error: Robot did not reach home position within the timeout.")
    return False

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. Argument Parsing
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Replay Neuracore datasets on the Piper robot.")
    parser.add_argument(
        "--dataset-name", type=str, required=True, 
        help="Name of the Neuracore dataset to load."
    )
    parser.add_argument(
        "--frequency", type=int, default=20, 
        help="Playback frequency in Hz. 0 plays the data aperiodically (default: 20)."
    )
    parser.add_argument(
        "--episode-index", type=int, default=0, 
        help="Episode index to replay (Frontend Number - 1). -1 plays all sequentially (default: 0)."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PIPER DATASET REPLAY".center(60))
    print("=" * 60)
    print("⚠️  WARNING: Robot will move identically to the recorded trajectory.")
    print("⚠️  Be ready to press the emergency stop or Ctrl+C.")
    if args.frequency == 0:
        print("⏱️  Playback set to APERIODIC (Frequency = 0).")
    else:
        print(f"⏱️  Playback synchronized at {args.frequency} Hz.")
    print("=" * 60)

    # ---------------------------------------------------------
    # 2. Hardware Initialization
    # ---------------------------------------------------------
    print("\n🤖 Initializing Piper robot controller...")
    robot_controller = PiperController(
        can_interface="can0", 
        robot_rate=ROBOT_RATE,
        control_mode=PiperController.ControlMode.JOINT_SPACE,
        neutral_joint_angles=NEUTRAL_JOINT_ANGLES, 
        debug_mode=False,
    )
    robot_controller.start_control_loop()

    # ---------------------------------------------------------
    # 3. Neuracore Connection & Dataset Synchronization
    # ---------------------------------------------------------
    print("\n🔑 Logging in to Neuracore...")
    nc.login()

    input_mods = [DataType.JOINT_POSITIONS, DataType.RGB_IMAGES, DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS]
    output_mods = [DataType.JOINT_TARGET_POSITIONS, DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS]
    
    synced_dataset = load_and_sync_dataset(
        args.dataset_name, 
        args.frequency, 
        input_mods, 
        output_mods
    )

    if len(synced_dataset) == 0:
        print("❌ Error: The synchronized dataset is empty. Exiting.")
        robot_controller.stop_control_loop()
        robot_controller.cleanup()
        sys.exit(1)

    if args.episode_index == -1:
        episode_indices = list(range(len(synced_dataset)))
        print(f"\n📊 Found {len(synced_dataset)} episodes. Will play all sequentially.")
    else:
        if args.episode_index >= len(synced_dataset) or args.episode_index < 0:
            print(f"❌ Error: Index {args.episode_index} is out of bounds (Dataset has {len(synced_dataset)} episodes).")
            print("💡 Reminder: If the frontend says episode '3', you must request index '2'.")
            robot_controller.stop_control_loop()
            robot_controller.cleanup()
            sys.exit(1)
            
        episode_indices = [args.episode_index]
        print(f"\n📊 Playing episode {args.episode_index} only.")

    # ---------------------------------------------------------
    # 4. Playback Loop
    # ---------------------------------------------------------
    try:
        for episode_idx in episode_indices:
            episode = synced_dataset[episode_idx]
            
            if len(episode) == 0:
                print(f"⚠️ Warning: Episode {episode_idx} is empty. Skipping...")
                continue

            if not wait_for_home(robot_controller):
                print("⚠️ Skipping episode due to homing failure.")
                continue

            print(f"\n{'='*60}\n🎬 Extracting Episode {episode_idx} / {len(synced_dataset) - 1}\n{'='*60}")

            rgb_frames_per_step = []
            parallel_gripper_open_amounts = []
            joint_positions = []

            # Pre-compute and Unpack
            for step in tqdm(episode, desc="Unpacking frames into memory"):
                step = cast(SynchronizedPoint, step)
                
                j_data = step.data.get(DataType.JOINT_TARGET_POSITIONS, {})
                if all(jn in j_data for jn in JOINT_NAMES):
                    joint_positions.append([j_data[jn].value for jn in JOINT_NAMES])
                else:
                    if len(joint_positions) > 0:
                        joint_positions.append(joint_positions[-1])
                    else:
                        continue 

                g_data = step.data.get(DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS, {})
                parallel_gripper_open_amounts.append(
                    g_data[GRIPPER_NAME].open_amount if GRIPPER_NAME in g_data else 0.0
                )

                rgb_data = step.data.get(DataType.RGB_IMAGES, {})
                rgb_frames_per_step.append({c: img.frame for c, img in rgb_data.items()})

            if not joint_positions:
                print(f"⚠️ Warning: No valid joint data found in episode {episode_idx}. Skipping...")
                continue

            joint_positions = np.degrees(np.array(joint_positions))

            # Hardware Execution
            print(f"\n🚀 Replaying episode {episode_idx} on hardware...")
            for idx in tqdm(range(len(joint_positions)), desc="Replaying"):
                start_time = time.time()
                
                robot_controller.set_target_joint_angles(joint_positions[idx])
                robot_controller.set_gripper_open_value(parallel_gripper_open_amounts[idx])

                if idx < len(rgb_frames_per_step):
                    for cam_name, frame_rgb in rgb_frames_per_step[idx].items():
                        frame_bgr = cv2.cvtColor(np.asarray(frame_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
                        cv2.imshow(f"Replay: {cam_name}", frame_bgr)
                        
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n🛑 'q' pressed, skipping remainder of replay...")
                    break

                # Handle Frequency Timing (0 = aperiodic, >0 = synchronized)
                if args.frequency > 0:
                    elapsed = time.time() - start_time
                    time.sleep(max(0.0, (1.0 / args.frequency) - elapsed))
            
            cv2.destroyAllWindows()
            
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt detected. Halting playback...")
        print("⚠️  Gracefully disabling the robot. Power to motors will be cut off.")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ Unhandled exception during playback: {e}")
        cv2.destroyAllWindows()

    # ---------------------------------------------------------
    # 5. Graceful Teardown & Cleanup
    # ---------------------------------------------------------
    print("\n🧹 Cleaning up subsystems...")
    # Triggering the cleanup gracefully disables the robot (cutting power securely)
    robot_controller.stop_control_loop()
    robot_controller.cleanup()
    print("👋 Replay complete.")