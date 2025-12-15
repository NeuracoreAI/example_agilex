#!/usr/bin/env python3
"""Minimal Piper Robot Policy Test - Terminal only, no GUI.

Simple script that:
1. Enables robot
2. Sends robot home
3. Runs policy in continuous loop (get image, run policy, execute horizon, repeat)
4. On cancellation: sends robot home and exits
"""

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path

import neuracore as nc
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import (
    GRIPPER_LOGGING_NAME,
    JOINT_NAMES,
    NEUTRAL_JOINT_ANGLES,
    POLICY_EXECUTION_RATE,
    PREDICTION_HORIZON_EXECUTION_RATIO,
    ROBOT_RATE,
    URDF_PATH,
)
from common.data_manager import DataManager, RobotActivityState
from common.policy_state import PolicyState
from common.threads.camera import camera_thread
from common.threads.joint_state import joint_state_thread

from piper_controller import PiperController


def run_policy(
    data_manager: DataManager,
    policy: nc.policy,
    policy_state: PolicyState,
) -> bool:
    """Run policy and get prediction horizon."""
    # Get current state
    current_joint_angles = data_manager.get_current_joint_angles()
    if current_joint_angles is None:
        print("⚠️  No joint angles available")
        return False

    gripper_open_value = data_manager.get_current_gripper_open_value()
    if gripper_open_value is None:
        print("⚠️  No gripper value available")
        return False

    rgb_image = data_manager.get_rgb_image()
    if rgb_image is None:
        print("⚠️  No RGB image available")
        return False

    # Get current gripper open value
    gripper_open_value = data_manager.get_current_gripper_open_value()
    if gripper_open_value is None:
        print("⚠️  No gripper open value available")
        return False

    # Get current RGB image
    rgb_image = data_manager.get_rgb_image()
    if rgb_image is None:
        print("⚠️  No RGB image available")
        return False

    # Prepare data for NeuraCore logging
    joint_angles_rad = np.radians(current_joint_angles)
    joint_positions_dict = {
        JOINT_NAMES[i]: angle for i, angle in enumerate(joint_angles_rad)
    }
    gripper_open_amounts_dict = {GRIPPER_LOGGING_NAME: gripper_open_value}

    # Log joint positions parallel gripper open amounts and RGB image to NeuraCore
    nc.log_joint_positions(joint_positions_dict)
    nc.log_gripper_data(open_amounts=gripper_open_amounts_dict)
    nc.log_rgb("camera", rgb_image)

    # timestamp = time.time()
    # sync_point = SyncPoint(
    #     joint_positions=JointData(values=joint_positions_dict, timestamp=timestamp),
    #     parallel_gripper_open_amounts=ParallelGripperOpenAmountData(
    #         open_amounts=gripper_open_amounts_dict, timestamp=timestamp
    #     ),
    #     rgb_images={"camera": CameraData(frame=rgb_image, timestamp=timestamp)},
    #     timestamp=timestamp,
    # )

    # Get policy prediction
    try:
        start_time = time.time()
        predicted_sync_points = policy.predict(timeout=5)
        elapsed = time.time() - start_time
        print(f"✓ Got {len(predicted_sync_points)} actions in {elapsed:.3f}s")

        # Save full horizon and set execution ratio (clipping occurs on lock)
        policy_state.set_execution_ratio(PREDICTION_HORIZON_EXECUTION_RATIO)
        policy_state.set_prediction_horizon_sync_points(predicted_sync_points)
        return True

    except Exception as e:
        print(f"✗ Policy prediction failed: {e}")
        traceback.print_exc()
        return False


def execute_horizon(
    data_manager: DataManager,
    policy_state: PolicyState,
    robot_controller: PiperController,
) -> None:
    """Execute prediction horizon."""
    policy_state.start_policy_execution()
    data_manager.set_robot_activity_state(RobotActivityState.POLICY_CONTROLLED)

    locked_horizon_sync_points = (
        policy_state.get_locked_prediction_horizon_sync_points()
    )
    horizon_length = policy_state.get_locked_prediction_horizon_length()
    dt = 1.0 / POLICY_EXECUTION_RATE

    for i in range(horizon_length):
        sync_point = locked_horizon_sync_points[i]
        if sync_point.joint_target_positions is not None:
            joint_targets_rad = sync_point.joint_target_positions.numpy(
                order=JOINT_NAMES
            )
            joint_targets_deg = np.degrees(joint_targets_rad)
            robot_controller.set_target_joint_angles(joint_targets_deg)

        if (
            sync_point.parallel_gripper_open_amounts is not None
            and GRIPPER_LOGGING_NAME
            in sync_point.parallel_gripper_open_amounts.open_amounts
        ):
            gripper_value = sync_point.parallel_gripper_open_amounts.open_amounts[
                GRIPPER_LOGGING_NAME
            ]
            robot_controller.set_gripper_open_value(gripper_value)

        # Sleep to maintain rate
        time.sleep(dt)

    # End execution
    policy_state.end_policy_execution()
    data_manager.set_robot_activity_state(RobotActivityState.ENABLED)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Piper Policy Test")
    parser.add_argument(
        "--model-file",
        type=str,
        required=True,
        help="Path to model file (.nc.zip)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PIPER POLICY ROLLOUT")
    print("=" * 60)

    # Initialize NeuraCore
    print("\n🔧 Initializing NeuraCore...")
    nc.login()
    nc.connect_robot(
        robot_name="AgileX PiPER",
        urdf_path=str(URDF_PATH),
        overwrite=False,
    )

    # Load policy
    print(f"\n🤖 Loading policy from: {args.model_file}...")
    policy = nc.policy(model_file=args.model_file, device="cuda")
    print("✓ Policy loaded")

    # Initialize state
    data_manager = DataManager()
    policy_state = PolicyState()

    # Initialize robot controller
    print("\n🤖 Initializing robot controller...")
    robot_controller = PiperController(
        can_interface="can0",
        robot_rate=ROBOT_RATE,
        control_mode=PiperController.ControlMode.JOINT_SPACE,
        neutral_joint_angles=NEUTRAL_JOINT_ANGLES,
        debug_mode=False,
    )
    robot_controller.start_control_loop()

    # Start joint state thread
    print("\n📊 Starting joint state thread...")
    joint_state_thread_obj = threading.Thread(
        target=joint_state_thread, args=(data_manager, robot_controller), daemon=True
    )
    joint_state_thread_obj.start()

    # Start camera thread
    print("\n📷 Starting camera thread...")
    camera_thread_obj = threading.Thread(
        target=camera_thread, args=(data_manager,), daemon=True
    )
    camera_thread_obj.start()

    # Wait for threads to initialize
    print("\n⏳ Waiting for initialization...")
    time.sleep(2.0)

    try:
        # Enable robot
        print("\n🟢 Enabling robot...")
        robot_controller.resume_robot()
        data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
        print("✓ Robot enabled")

        # Home robot
        print("\n🏠 Moving to home position...")
        robot_controller.move_to_home()
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)

        # Wait for homing to complete
        start_time = time.time()
        while (
            data_manager.get_robot_activity_state() == RobotActivityState.HOMING
            and not robot_controller.is_robot_homed()
            and time.time() - start_time < 5.0
        ):
            time.sleep(0.1)
        print("✓ Robot homed")

        # Policy execution loop
        print("\n🚀 Starting policy execution loop...")
        print("Press Ctrl+C to stop\n")

        while True:
            # Run policy
            if not run_policy(data_manager, policy, policy_state):
                print("⚠️  Policy run failed, retrying...")
                time.sleep(0.5)
                continue

            # Execute horizon
            execute_horizon(data_manager, policy_state, robot_controller)

    except KeyboardInterrupt:
        print("\n\n👋 Interrupt received - shutting down...")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")

        # Home robot
        print("\n🏠 Moving to home position...")
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
        robot_controller.move_to_home()

        # Wait for homing to complete
        start_time = time.time()
        while (
            data_manager.get_robot_activity_state() == RobotActivityState.HOMING
            and not robot_controller.is_robot_homed()
            and time.time() - start_time < 5.0
        ):
            time.sleep(0.1)
        print("✓ Robot homed")

        # Shutdown
        policy.disconnect()
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        data_manager.request_shutdown()
        joint_state_thread_obj.join()
        camera_thread_obj.join()
        time.sleep(0.5)  # Give threads time to stop

        robot_controller.cleanup()
        nc.logout()

        print("✓ Cleanup complete")
        print("\n👋 Done.")
