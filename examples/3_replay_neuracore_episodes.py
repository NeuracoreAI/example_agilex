#!/usr/bin/env python3
"""Replay a recorded Neuracore dataset on the Piper robot."""

import argparse
import sys
import time
from pathlib import Path
from typing import cast
import cv2
import neuracore as nc
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import GRIPPER_NAME, JOINT_NAMES, NEUTRAL_JOINT_ANGLES, ROBOT_RATE
from common.dataset_helpers import load_and_sync_dataset
from neuracore_types import DataType, SynchronizedPoint
from piper_controller import PiperController

def wait_for_home(robot_controller, timeout: int = 10):
    robot_controller.move_to_home()
    for _ in range(timeout):
        if robot_controller.is_robot_homed():
            print("✓ Robot is at home position.")
            return True
        time.sleep(1)
    print("❌ Robot did not reach home position.")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--frequency", type=int, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    args = parser.parse_args()

    print("\n🤖 Initializing Piper robot controller...")
    robot_controller = PiperController(
        can_interface="can0", robot_rate=ROBOT_RATE,
        control_mode=PiperController.ControlMode.JOINT_SPACE,
        neutral_joint_angles=NEUTRAL_JOINT_ANGLES, debug_mode=False,
    )
    robot_controller.start_control_loop()

    print("\n🔑 Logging in to Neuracore...")
    nc.login()

    input_mods = [DataType.JOINT_POSITIONS, DataType.RGB_IMAGES, DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS]
    output_mods = [DataType.JOINT_TARGET_POSITIONS, DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS]
    
    synced_dataset = load_and_sync_dataset(args.dataset_name, args.frequency, input_mods, output_mods)

    episode_indices = list(range(len(synced_dataset))) if args.episode_index == -1 else [args.episode_index]
    print(f"\n📊 Playing {len(episode_indices)} episode(s).")

    try:
        for episode_idx in episode_indices:
            if not wait_for_home(robot_controller): continue

            print(f"\n{'='*60}\n🎬 Playing Episode {episode_idx} / {len(synced_dataset) - 1}\n{'='*60}")
            episode = synced_dataset[episode_idx]

            rgb_frames_per_step, parallel_gripper_open_amounts, joint_positions = [], [], []

            for step in tqdm(episode, desc=f"Collecting data"):
                step = cast(SynchronizedPoint, step)
                
                # Extract Joints
                j_data = step.data.get(DataType.JOINT_TARGET_POSITIONS, {})
                joint_positions.append([j_data[jn].value for jn in JOINT_NAMES if jn in j_data])

                # Extract Gripper
                g_data = step.data.get(DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS, {})
                parallel_gripper_open_amounts.append(g_data[GRIPPER_NAME].open_amount if GRIPPER_NAME in g_data else 0.0)

                # Extract Camera
                rgb_data = step.data.get(DataType.RGB_IMAGES, {})
                rgb_frames_per_step.append({c: img.frame for c, img in rgb_data.items()})

            joint_positions = np.degrees(np.array(joint_positions))

            print(f"\n🚀 Replaying episode {episode_idx}...")
            for idx in tqdm(range(len(joint_positions)), desc="Replaying"):
                start_time = time.time()
                robot_controller.set_target_joint_angles(joint_positions[idx])
                robot_controller.set_gripper_open_value(parallel_gripper_open_amounts[idx])

                if idx < len(rgb_frames_per_step):
                    for cam_name, frame_rgb in rgb_frames_per_step[idx].items():
                        cv2.imshow(f"Replay: {cam_name}", cv2.cvtColor(np.asarray(frame_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n🛑 'q' pressed, stopping replay..."); break

                time.sleep(max(0, 1 / args.frequency - (time.time() - start_time)))
            
            cv2.destroyAllWindows()
            
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt detected.")
        cv2.destroyAllWindows()

    robot_controller.stop_control_loop()
    robot_controller.cleanup()