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
from neuracore_types import (
    BatchedJointData,
    BatchedParallelGripperOpenAmountData,
    DataType,
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import (
    CAMERA_LOGGING_NAME,
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


def convert_predictions_to_horizon_dict(predictions: dict) -> dict[str, list[float]]:
    """Convert predictions dict to horizon dict format."""
    horizon: dict[str, list[float]] = {}

    # Extract joint target positions
    if DataType.JOINT_TARGET_POSITIONS in predictions:
        joint_data = predictions[DataType.JOINT_TARGET_POSITIONS]
        for joint_name in JOINT_NAMES:
            if joint_name in joint_data:
                batched = joint_data[joint_name]
                if isinstance(batched, BatchedJointData):
                    # Extract values: (B, T, 1) -> list[float], taking B=0
                    values = batched.value[0, :, 0].cpu().numpy().tolist()
                    horizon[joint_name] = values

    # Extract gripper open amounts
    if DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in predictions:
        gripper_data = predictions[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS]
        if GRIPPER_LOGGING_NAME in gripper_data:
            batched = gripper_data[GRIPPER_LOGGING_NAME]
            if isinstance(batched, BatchedParallelGripperOpenAmountData):
                # Extract values: (B, T, 1) -> list[float], taking B=0
                values = batched.open_amount[0, :, 0].cpu().numpy().tolist()
                horizon[GRIPPER_LOGGING_NAME] = values

    return horizon


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

    # Get target gripper open value because this is how the policy was trained
    gripper_open_value = data_manager.get_target_gripper_open_value()
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

    # Log joint positions, parallel gripper open amounts, and RGB image to NeuraCore
    nc.log_joint_positions(joint_positions_dict)
    nc.log_parallel_gripper_open_amount(GRIPPER_LOGGING_NAME, gripper_open_value)
    nc.log_rgb(CAMERA_LOGGING_NAME, rgb_image)

    # Get policy prediction
    try:
        start_time = time.time()
        predictions = policy.predict(timeout=5)
        prediction_horizon = convert_predictions_to_horizon_dict(predictions)
        elapsed = time.time() - start_time

        # Get horizon length from the first joint (all should have same length)
        horizon_length = policy_state.get_prediction_horizon_length()
        print(f"✓ Got {horizon_length} actions in {elapsed:.3f}s")

        # Set execution ratio and save prediction horizon
        policy_state.set_execution_ratio(PREDICTION_HORIZON_EXECUTION_RATIO)
        policy_state.set_prediction_horizon(prediction_horizon)
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

    locked_horizon = policy_state.get_locked_prediction_horizon()
    horizon_length = policy_state.get_locked_prediction_horizon_length()
    dt = 1.0 / POLICY_EXECUTION_RATE

    for i in range(horizon_length):
        # Send current action to robot (if available)
        if all(joint_name in locked_horizon for joint_name in JOINT_NAMES):
            current_joint_target_positions_rad = np.array(
                [locked_horizon[joint_name][i] for joint_name in JOINT_NAMES]
            )
            current_joint_target_positions_deg = np.degrees(
                current_joint_target_positions_rad
            )
            robot_controller.set_target_joint_angles(current_joint_target_positions_deg)

        # Send current gripper open value to robot (if available)
        if GRIPPER_LOGGING_NAME in locked_horizon:
            current_gripper_open_value = locked_horizon[GRIPPER_LOGGING_NAME][i]
            robot_controller.set_gripper_open_value(current_gripper_open_value)

        # Sleep to maintain rate
        time.sleep(dt)

    # End execution
    policy_state.end_policy_execution()
    data_manager.set_robot_activity_state(RobotActivityState.ENABLED)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Piper Policy Test")
    parser.add_argument(
        "--train-run-name",
        type=str,
        default=None,
        help="Name of the training run to load policy from (for cloud training). Mutually exclusive with --model-path.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to local model file to load policy from. Mutually exclusive with --train-run-name.",
    )
    args = parser.parse_args()

    # Validate that exactly one of train-run-name or model-path is provided
    if (args.train_run_name is None) == (args.model_path is None):
        parser.error(
            "Exactly one of --train-run-name or --model-path must be provided (not both, not neither)"
        )

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
    # NOTE: The model_output_order MUST match the exact order used during training
    # This order is determined by the output_robot_data_spec in the training config.
    # The order here should match the order in your training config's output_robot_data_spec.
    model_input_order = {
        DataType.JOINT_POSITIONS: JOINT_NAMES,
        DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: [GRIPPER_LOGGING_NAME],
        DataType.RGB_IMAGES: [CAMERA_LOGGING_NAME],
    }
    model_output_order = {
        DataType.JOINT_TARGET_POSITIONS: JOINT_NAMES,
        DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: [GRIPPER_LOGGING_NAME],
    }

    print("\n📋 Model input order:")
    for data_type, names in model_input_order.items():
        print(f"  {data_type.name}: {names}")
    print("\n📋 Model output order:")
    for data_type, names in model_output_order.items():
        print(f"  {data_type.name}: {names}")

    if args.train_run_name is not None:
        print(f"\n🤖 Loading policy from training run: {args.train_run_name}...")
        policy = nc.policy(
            train_run_name=args.train_run_name,
            model_input_order=model_input_order,
            model_output_order=model_output_order,
        )
    else:
        print(f"\n🤖 Loading policy from model file: {args.model_path}...")
        policy = nc.policy(
            model_file=args.model_path,
            device="cuda",
            model_input_order=model_input_order,
            model_output_order=model_output_order,
        )
    print("  ✓ Policy loaded successfully")

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
