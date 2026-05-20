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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import (
    NEUTRAL_JOINT_ANGLES,
    POLICY_EXECUTION_RATE,
    PREDICTION_HORIZON_EXECUTION_RATIO,
    ROBOT_RATE,
    URDF_PATH,
)
from common.data_manager import DataManager, RobotActivityState
from common.policy_helpers import (
    convert_predictions_to_horizon,
    embodiment_names_ordered,
    get_policy_embodiments,
    gripper_open_at_index,
    joint_targets_deg_at_index,
    log_robot_state_for_policy,
    print_policy_embodiments,
)
from common.policy_state import PolicyState
from common.threads.joint_state import joint_state_thread
from common.threads.realsense_camera import camera_thread
from neuracore_types import DataType, EmbodimentDescription

from piper_controller import PiperController


def run_policy(
    data_manager: DataManager,
    policy: nc.policy,
    policy_state: PolicyState,
    input_embodiment_description: EmbodimentDescription,
) -> bool:
    """Run policy and get prediction horizon."""
    if not log_robot_state_for_policy(data_manager, input_embodiment_description):
        print("⚠️  No policy input data available")
        return False

    # Get policy prediction
    try:
        start_time = time.time()
        predictions = policy.predict(timeout=5)
        prediction_horizon = convert_predictions_to_horizon(predictions)
        elapsed = time.time() - start_time

        # Get horizon length from the first joint (all should have same length)
        horizon_length = policy_state.get_prediction_horizon_length()
        print(f"✓ Got {horizon_length} actions in {elapsed:.3f}s")

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
    frequency: int,
    input_embodiment_description: EmbodimentDescription,
    output_gripper_names: list[str] | None,
) -> None:
    """Execute prediction horizon."""
    policy_state.start_policy_execution()
    data_manager.set_robot_activity_state(RobotActivityState.POLICY_CONTROLLED)

    locked_horizon = policy_state.get_locked_prediction_horizon()
    horizon_length = policy_state.get_locked_prediction_horizon_length()
    dt = 1.0 / frequency

    for i in range(horizon_length):
        start_time = time.time()

        joint_targets_deg = joint_targets_deg_at_index(locked_horizon, i)
        if joint_targets_deg is not None:
            robot_controller.set_target_joint_angles(joint_targets_deg)

        gripper_target = gripper_open_at_index(
            locked_horizon, i, gripper_names=output_gripper_names
        )
        if gripper_target is not None:
            robot_controller.set_gripper_open_value(gripper_target)

        log_robot_state_for_policy(data_manager, input_embodiment_description)

        # Sleep to maintain rate
        elapsed = time.time() - start_time
        time.sleep(max(0, dt - elapsed))

    # End execution
    policy_state.end_policy_execution()
    data_manager.set_robot_activity_state(RobotActivityState.ENABLED)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Piper Policy Test")
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument(
        "--train-run-name",
        type=str,
        default=None,
        help="Name of the training run to load policy from (for cloud training).",
    )
    policy_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to local model file to load policy from.",
    )
    policy_group.add_argument(
        "--remote-endpoint-name",
        type=str,
        default=None,
        help="Name of remote Neuracore policy endpoint.",
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default="AgileX PiPER",
        help="Neuracore robot name (policy embodiment resolution).",
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=POLICY_EXECUTION_RATE,
        help="Frequency of policy execution",
    )
    parser.add_argument(
        "--execution-ratio",
        type=float,
        default=PREDICTION_HORIZON_EXECUTION_RATIO,
        help="Execution ratio of the policy",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PIPER POLICY ROLLOUT")
    print("=" * 60)

    # Initialize Neuracore
    print("\n🔧 Initializing Neuracore...")
    nc.login()
    nc.connect_robot(
        robot_name=args.robot_name,
        urdf_path=str(URDF_PATH),
        overwrite=False,
    )

    if args.remote_endpoint_name is not None:
        print(
            f"\n🤖 Connecting to remote policy endpoint: {args.remote_endpoint_name}..."
        )
        try:
            policy = nc.policy_remote_server(args.remote_endpoint_name)
        except nc.EndpointError:
            print(
                f"❌ Endpoint '{args.remote_endpoint_name}' not available. "
                "Please start it from the Neuracore dashboard."
            )
            sys.exit(1)
    elif args.train_run_name is not None:
        print(f"\n🤖 Loading policy from training run: {args.train_run_name}...")
        policy = nc.policy(
            train_run_name=args.train_run_name,
            device="cuda",
            robot_name=args.robot_name,
        )
    else:
        print(f"\n🤖 Loading policy from model file: {args.model_path}...")
        policy = nc.policy(
            model_file=args.model_path,
            device="cuda",
            robot_name=args.robot_name,
        )
    print("  ✓ Policy loaded successfully")
    input_embodiment_description, output_embodiment_description = (
        get_policy_embodiments(policy)
    )
    print_policy_embodiments(
        input_embodiment_description, output_embodiment_description
    )
    output_gripper_names = None
    if output_embodiment_description is not None:
        gripper_spec = output_embodiment_description.get(
            DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS
        )
        if gripper_spec is not None:
            output_gripper_names = embodiment_names_ordered(gripper_spec)

    # Initialize state
    data_manager = DataManager()
    policy_state = PolicyState()
    policy_state.set_execution_ratio(args.execution_ratio)

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
            if not run_policy(
                data_manager, policy, policy_state, input_embodiment_description
            ):
                print("⚠️  Policy run failed, retrying...")
                time.sleep(0.5)
                continue

            # Execute horizon
            execute_horizon(
                data_manager,
                policy_state,
                robot_controller,
                args.frequency,
                input_embodiment_description,
                output_gripper_names,
            )

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
