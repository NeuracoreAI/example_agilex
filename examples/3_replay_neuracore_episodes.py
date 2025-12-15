"""Replay a recorded Neuracore dataset on the Piper robot."""

import argparse
import sys
import time
from pathlib import Path

import neuracore as nc
import numpy as np
from common.configs import NEUTRAL_JOINT_ANGLES, ROBOT_RATE
from neuracore_types import DataType
from tqdm import tqdm

# Add parent directory to path to piper_controller
sys.path.insert(0, str(Path(__file__).parent.parent))

from piper_controller import PiperController


def main() -> None:
    """Main function for replaying a Neuracore dataset on the Piper robot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--frequency", type=int, required=False, default=1)
    parser.add_argument("--episode-index", type=int, required=False, default=0)
    args = parser.parse_args()

    # Initialize robot controller
    print("\n🤖 Initializing Piper robot controller...")
    robot_controller = PiperController(
        can_interface="can0",
        robot_rate=ROBOT_RATE,
        control_mode=PiperController.ControlMode.JOINT_SPACE,
        neutral_joint_angles=NEUTRAL_JOINT_ANGLES,
        debug_mode=False,
    )
    # Start robot control loop
    print("\n🚀 Starting robot control loop...")
    robot_controller.start_control_loop()

    # Login to Neuracore
    print("\n🔑 Logging in to Neuracore...")
    nc.login()

    # Get dataset from Neuracore
    print("\n🔍 Getting dataset from Neuracore...")
    dataset = nc.get_dataset(args.dataset_name)

    # Synchronize dataset
    print("\n🔁 Synchronizing dataset...")
    synced_dataset = dataset.synchronize(
        frequency=args.frequency,
        data_types=[
            DataType.JOINT_POSITIONS,
            DataType.RGB_IMAGE,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
        ],
        prefetch_videos=True,
        max_prefetch_workers=2,
    )

    # Determine which episodes to play
    episode_indices: list[int] = []
    if args.episode_index == -1:
        episode_indices = list(range(len(synced_dataset)))
        print(f"\n📊 Found {len(synced_dataset)} episodes. Will play all episodes.")
    else:
        episode_indices = [args.episode_index]
        print(f"\n📊 Playing episode {args.episode_index} only.")

    # Play episodes
    try:
        for episode_idx in episode_indices:

            robot_controller.move_to_home()
            seconds_to_wait = 10
            while not robot_controller.is_robot_homed():
                time.sleep(1)
                seconds_to_wait -= 1
                if seconds_to_wait <= 0:
                    break
                print(
                    f"🔍 Waiting for robot to reach home position... {seconds_to_wait} seconds remaining."
                )
            if robot_controller.is_robot_homed():
                print("✓ Robot is at home position.")
            else:
                print("❌ Robot did not reach home position within 10 seconds.")
                print(
                    f"🔍 Current joint angles: {robot_controller.get_current_joint_angles()}"
                )
                print(f"🔍 Home joint angles: {robot_controller.HOME_JOINT_ANGLES}")

            print(f"\n{'='*60}")
            print(f"🎬 Playing Episode {episode_idx} / {len(synced_dataset) - 1}")
            print(f"{'='*60}")

            episode = synced_dataset[episode_idx]

            print(f"\n🚀 Collecting episode {episode_idx} data...")
            rgb_images = []
            parallel_gripper_open_amounts = []
            joint_positions = []
            for step in tqdm(episode, desc=f"Collecting episode {episode_idx}"):
                joint_positions.append(list(step.joint_positions.values.values()))
                parallel_gripper_open_amounts.append(
                    step.parallel_gripper_open_amounts.open_amounts["gripper"]
                )
                if step.rgb_images is not None:
                    for _, cam_data in step.rgb_images.items():
                        rgb_images.append(cam_data.frame)
                        break

            joint_positions = np.degrees(np.array(joint_positions))
            parallel_gripper_open_amounts = np.array(parallel_gripper_open_amounts)

            print(f"\n🚀 Replaying episode {episode_idx} data...")
            for index in tqdm(
                range(len(joint_positions)), desc=f"Replaying episode {episode_idx}"
            ):
                start_time = time.time()
                robot_controller.set_target_joint_angles(joint_positions[index])
                robot_controller.set_gripper_open_value(
                    parallel_gripper_open_amounts[index]
                )
                end_time = time.time()
                time.sleep(max(0, 1 / args.frequency - (end_time - start_time)))
            print(f"🎉 Episode {episode_idx} replay completed.")

        if args.episode_index == -1:
            print(f"\n{'='*60}")
            print(f"🎉 All {len(synced_dataset)} episodes replay completed!")
            print(f"{'='*60}")
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt detected, stopping robot control loop...")

    robot_controller.stop_control_loop()
    robot_controller.cleanup()


if __name__ == "__main__":
    main()
