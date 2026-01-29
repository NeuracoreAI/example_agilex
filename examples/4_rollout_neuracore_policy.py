#!/usr/bin/env python3
"""Piper Robot Test with Neuracore policy.

This script loads a trained Neuracore policy, reads status from the piper robot
controlled by the Meta Quest controller, and replays the prediction horizon virtually
on Viser to test the stability of the policy output.
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

# Add parent directory to path to import pink_ik_solver and piper_controller
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import (
    CAMERA_FRAME_STREAMING_RATE,
    CAMERA_LOGGING_NAME,
    CONTROLLER_BETA,
    CONTROLLER_D_CUTOFF,
    CONTROLLER_DATA_RATE,
    CONTROLLER_MIN_CUTOFF,
    DAMPING_COST,
    FRAME_TASK_GAIN,
    GRIPPER_FRAME_NAME,
    GRIPPER_LOGGING_NAME,
    IK_SOLVER_RATE,
    JOINT_NAMES,
    JOINT_STATE_STREAMING_RATE,
    LM_DAMPING,
    MAX_ACTION_ERROR_THRESHOLD,
    MAX_SAFETY_THRESHOLD,
    NEUTRAL_JOINT_ANGLES,
    ORIENTATION_COST,
    POLICY_EXECUTION_RATE,
    POSITION_COST,
    POSTURE_COST_VECTOR,
    PREDICTION_HORIZON_EXECUTION_RATIO,
    ROBOT_RATE,
    SOLVER_DAMPING_VALUE,
    SOLVER_NAME,
    TARGETING_POSE_TIME_THRESHOLD,
    URDF_PATH,
    VISUALIZATION_RATE,
)
from common.data_manager import DataManager, RobotActivityState
from common.policy_state import PolicyState
from common.robot_visualizer import RobotVisualizer
from common.threads.camera import camera_thread
from common.threads.ik_solver import ik_solver_thread
from common.threads.joint_state import joint_state_thread
from common.threads.quest_reader import quest_reader_thread
from meta_quest_teleop.reader import MetaQuestReader

from pink_ik_solver import PinkIKSolver
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
    if DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS in predictions:
        gripper_data = predictions[DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS]
        if GRIPPER_LOGGING_NAME in gripper_data:
            batched = gripper_data[GRIPPER_LOGGING_NAME]
            if isinstance(batched, BatchedParallelGripperOpenAmountData):
                # Extract values: (B, T, 1) -> list[float], taking B=0
                values = batched.open_amount[0, :, 0].cpu().numpy().tolist()
                horizon[GRIPPER_LOGGING_NAME] = values

    return horizon


def toggle_robot_enabled_status(
    data_manager: DataManager,
    robot_controller: PiperController,
    visualizer: RobotVisualizer,
) -> None:
    """Handle Button A press to toggle robot enable/disable state."""
    robot_activity_state = data_manager.get_robot_activity_state()
    if robot_activity_state == RobotActivityState.ENABLED:
        # Disable robot
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        robot_controller.graceful_stop()
        # Reset teleop state when disabling robot
        data_manager.set_teleop_state(False, None, None)
        visualizer.update_toggle_robot_enabled_status(False)
        print("✓ 🔴 Robot disabled (Button A)")
    elif robot_activity_state == RobotActivityState.DISABLED:
        if robot_controller.resume_robot():
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
            visualizer.update_toggle_robot_enabled_status(True)
            print("✓ 🟢 Robot enabled (Button A)")
        else:
            print("✗ Failed to enable robot")


def home_robot(data_manager: DataManager, robot_controller: PiperController) -> None:
    """Handle Button B press to move robot to home position."""
    robot_activity_state = data_manager.get_robot_activity_state()
    if robot_activity_state == RobotActivityState.ENABLED:
        print("🏠 Button B pressed - Moving to home position...")
        # Set state to HOMING to prevent IK thread from sending robot commands
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
        # Disable teleop during homing
        data_manager.set_teleop_state(False, None, None)
        ok = robot_controller.move_to_home()
        if not ok:
            print("✗ Failed to initiate home move")
            # Revert to ENABLED on failure
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
    else:
        print("⚠️  Button B pressed but robot is not enabled")


def run_policy(
    data_manager: DataManager,
    policy: nc.policy,
    policy_state: PolicyState,
    visualizer: RobotVisualizer,
    model_input_order: dict[DataType, list[str]],
) -> bool:
    """Handle Run Policy button press to capture state and get policy prediction."""
    print("Running policy...")

    # Get available data from data_manager (only log what the model expects)
    current_joint_angles = None
    gripper_open_value = None
    rgb_image = None

    # Only log data types that are in model_input_order
    if DataType.JOINT_POSITIONS in model_input_order:
        current_joint_angles = data_manager.get_current_joint_angles()
        if current_joint_angles is not None:
            joint_angles_rad = np.radians(current_joint_angles)
            joint_positions_dict = {
                joint_name: angle
                for joint_name, angle in zip(JOINT_NAMES, joint_angles_rad)
            }
            nc.log_joint_positions(joint_positions_dict)
            print("  ✓ Logged joint positions")
        else:
            print("  ⚠️  No current joint angles available")

    if DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in model_input_order:
        gripper_open_value = data_manager.get_current_gripper_open_value()
        if gripper_open_value is not None:
            nc.log_parallel_gripper_open_amount(
                GRIPPER_LOGGING_NAME, gripper_open_value
            )
            print("  ✓ Logged gripper open amount")
        else:
            print("  ⚠️  No gripper open value available")

    if DataType.RGB_IMAGES in model_input_order:
        rgb_image = data_manager.get_rgb_image()
        if rgb_image is not None:
            nc.log_rgb(CAMERA_LOGGING_NAME, rgb_image)
            print("  ✓ Logged RGB image")
        else:
            print("  ⚠️  No RGB image available")

    # Check if we have at least some data to run the policy
    if (
        current_joint_angles is None
        and gripper_open_value is None
        and rgb_image is None
    ):
        print("✗ No data available to run policy")
        return False

    # Get policy prediction
    try:
        start_time = time.time()
        predictions = policy.predict(timeout=5)
        prediction_horizon = convert_predictions_to_horizon_dict(predictions)
        end_time = time.time()
        horizon_length = policy_state.get_prediction_horizon_length()
        print(
            f"  ✓ Got {horizon_length} actions in {end_time - start_time:.3f} seconds"
        )

        prediction_ratio = visualizer.get_prediction_ratio()
        policy_state.set_execution_ratio(prediction_ratio)

        # Set policy inputs (only if available)
        if rgb_image is not None:
            policy_state.set_policy_rgb_image_input(rgb_image)
        if current_joint_angles is not None:
            policy_state.set_policy_state_input(current_joint_angles)

        # Store prediction horizon actions in policy state
        policy_state.set_prediction_horizon(prediction_horizon)

        visualizer.update_ghost_robot_visibility(True)
        policy_state.set_ghost_robot_playing(True)
        policy_state.reset_ghost_action_index()

    except Exception as e:
        print(f"✗ Failed to get policy prediction: {e}")
        traceback.print_exc()
        return False

    return True


def start_policy_execution(
    data_manager: DataManager, policy_state: PolicyState
) -> bool:
    """Handle Execute Policy button press to start policy execution."""
    print("Starting policy execution...")

    # Check if policy execution is already active
    if (
        data_manager.get_robot_activity_state() == RobotActivityState.POLICY_CONTROLLED
        and not policy_state.get_continuous_play_active()
    ):
        print("⚠️  Policy execution already in progress")
        return False
    # Check if robot is enabled
    elif data_manager.get_robot_activity_state() == RobotActivityState.DISABLED:
        print("⚠️  Cannot execute policy: Robot is disabled")
        return False

    # Get prediction horizon
    prediction_horizon = policy_state.get_prediction_horizon()
    prediction_horizon_length = policy_state.get_prediction_horizon_length()
    if prediction_horizon_length == 0:
        print("⚠️  No prediction horizon available. Make sure policy was run first.")
        return False

    # Check that we have joint data for all joints
    if not all(joint_name in prediction_horizon for joint_name in JOINT_NAMES):
        print("⚠️  First prediction in horizon has no joint targets")
        return False

    # Safety check: verify robot is close enough to first action
    current_joint_angles = data_manager.get_current_joint_angles()
    if current_joint_angles is None:
        print("⚠️  Cannot execute policy: No current joint angles available")
        return False
    # Get first action from horizon (index 0 for each joint)
    current_joint_target_positions_rad = np.array(
        [prediction_horizon[joint_name][0] for joint_name in JOINT_NAMES]
    )
    joint_differences = np.abs(
        current_joint_angles - np.degrees(current_joint_target_positions_rad)
    )
    if np.any(joint_differences > MAX_SAFETY_THRESHOLD):
        print("⚠️ Cannot execute policy: Robot too far from first action")
        print(f"   Differences: {[f'{d:.3f}' for d in joint_differences]}")
        print(f"   Threshold: {MAX_SAFETY_THRESHOLD}°")
        return False

    # All checks passed - start execution

    # Stop ghost visualization
    policy_state.set_ghost_robot_playing(False)

    # Deactivate teleop
    data_manager.set_teleop_state(False, None, None)

    # Lock policy inputs and start execution
    policy_state.start_policy_execution()

    # Verify locked horizon was created successfully
    locked_horizon_length = policy_state.get_locked_prediction_horizon_length()
    if locked_horizon_length == 0:
        print("⚠️  Failed to lock prediction horizon - horizon is empty")
        policy_state.end_policy_execution()
        return False

    print(f"✓ Starting policy execution with {locked_horizon_length} actions")

    # Change robot state to POLICY_CONTROLLED
    data_manager.set_robot_activity_state(RobotActivityState.POLICY_CONTROLLED)

    return True


def run_and_start_policy_execution(
    data_manager: DataManager,
    policy: nc.policy,
    policy_state: PolicyState,
    visualizer: RobotVisualizer,
    model_input_order: dict[DataType, list[str]],
) -> None:
    """Handle Run and Execute Policy button press to capture state, get policy prediction, and immediately execute it."""
    print("Run and Execute Policy for one prediction horizon")
    run_policy(data_manager, policy, policy_state, visualizer, model_input_order)
    start_policy_execution(data_manager, policy_state)


def end_policy_play(
    data_manager: DataManager,
    policy_state: PolicyState,
    visualizer: RobotVisualizer,
    policy_status_message: str,
) -> None:
    """End continuous play and set robot activity state to ENABLED and update policy status."""
    if policy_state.get_continuous_play_active():
        policy_state.set_continuous_play_active(False)
    visualizer.update_play_policy_button_status(False)
    policy_state.end_policy_execution()
    data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
    data_manager.set_teleop_state(False, None, None)
    visualizer.update_policy_status(policy_status_message)


def play_policy(
    data_manager: DataManager,
    policy: nc.policy,
    policy_state: PolicyState,
    visualizer: RobotVisualizer,
    model_input_order: dict[DataType, list[str]],
) -> None:
    """Handle Play Policy button press to start/stop continuous policy execution."""
    if not policy_state.get_continuous_play_active():
        # Start continuous play
        print("▶️  Play Policy button pressed - Starting continuous policy execution...")

        # Run policy to get prediction horizon
        success = run_policy(
            data_manager, policy, policy_state, visualizer, model_input_order
        )
        if not success:
            print("⚠️  Failed to run policy")
            end_policy_play(
                data_manager,
                policy_state,
                visualizer,
                "Continuous play stopped - prediction failed",
            )
            return

        # Execute policy
        success = start_policy_execution(data_manager, policy_state)
        if not success:
            print("⚠️  Failed to execute policy")
            end_policy_play(
                data_manager,
                policy_state,
                visualizer,
                "Continuous play stopped - execution failed",
            )
            return

        policy_state.set_continuous_play_active(True)
        visualizer.update_play_policy_button_status(True)

    else:
        # Stop continuous play
        print("⏹️  Stop Policy button pressed - Stopping continuous policy execution...")
        policy_state.set_continuous_play_active(False)
        end_policy_play(
            data_manager, policy_state, visualizer, "Policy execution stopped "
        )

        print("✓ Policy execution stopped and robot enabled")


def policy_execution_thread(
    policy: nc.policy,
    data_manager: DataManager,
    policy_state: PolicyState,
    robot_controller: PiperController,
    visualizer: RobotVisualizer,
    model_input_order: dict[DataType, list[str]],
) -> None:
    """Policy execution thread."""
    dt_execution = 1.0 / POLICY_EXECUTION_RATE
    # Throttle visualization updates to ~30Hz to avoid overwhelming Viser
    last_visualization_update = 0.0
    visualization_update_interval = 1.0 / 30.0  # 30 Hz
    while True:
        start_time = time.time()

        if (
            data_manager.get_robot_activity_state()
            == RobotActivityState.POLICY_CONTROLLED
        ):
            locked_horizon = policy_state.get_locked_prediction_horizon()
            execution_index = policy_state.get_execution_action_index()
            locked_horizon_length = policy_state.get_locked_prediction_horizon_length()

            # Debug output on first execution step
            if execution_index == 0 and locked_horizon_length > 0:
                print(
                    f"🔄 Policy execution started: {locked_horizon_length} actions, "
                    f"robot enabled: {robot_controller.is_robot_enabled()}"
                )

            if execution_index < locked_horizon_length:
                # Check if previous goal was achieved, if any
                current_joint_angles = data_manager.get_current_joint_angles()
                if (
                    execution_index > 0
                    and current_joint_angles is not None
                    and policy_state.get_execution_mode()
                    == PolicyState.ExecutionMode.TARGETING_POSE
                ):
                    targeting_pose_start_time = time.time()
                    while (
                        time.time() - targeting_pose_start_time
                        < TARGETING_POSE_TIME_THRESHOLD
                    ):
                        # Get previous action from horizon
                        if not all(
                            joint_name in locked_horizon for joint_name in JOINT_NAMES
                        ):
                            break
                        previous_joint_target_positions_rad = np.array(
                            [
                                locked_horizon[joint_name][execution_index - 1]
                                for joint_name in JOINT_NAMES
                            ]
                        )
                        previous_joint_target_positions_deg = np.degrees(
                            previous_joint_target_positions_rad
                        )
                        joint_errors = np.abs(
                            current_joint_angles - previous_joint_target_positions_deg
                        )
                        if np.any(joint_errors <= MAX_ACTION_ERROR_THRESHOLD):
                            break
                        time.sleep(0.001)

                # Send current action to robot (if available)
                if all(joint_name in locked_horizon for joint_name in JOINT_NAMES):
                    current_joint_target_positions_rad = np.array(
                        [
                            locked_horizon[joint_name][execution_index]
                            for joint_name in JOINT_NAMES
                        ]
                    )
                    current_joint_target_positions_deg = np.degrees(
                        current_joint_target_positions_rad
                    )
                    # Update data_manager with target joint angles for visualization
                    data_manager.set_target_joint_angles(
                        current_joint_target_positions_deg
                    )

                    # Verify robot controller is enabled before sending commands
                    if robot_controller.is_robot_enabled():
                        robot_controller.set_target_joint_angles(
                            current_joint_target_positions_deg
                        )
                    else:
                        print(
                            f"⚠️  Robot controller not enabled, skipping command at index {execution_index}"
                        )

                # Send current gripper open value to robot (if available)
                if GRIPPER_LOGGING_NAME in locked_horizon:
                    current_gripper_target_open_value = locked_horizon[
                        GRIPPER_LOGGING_NAME
                    ][execution_index]
                    robot_controller.set_gripper_open_value(
                        current_gripper_target_open_value
                    )

                # Update execution index
                policy_state.increment_execution_action_index()

                # Update status
                visualizer.update_policy_status(
                    f"Executing policy: {execution_index + 1}/{locked_horizon_length}"
                )
            # Check if continuous play is active
            elif policy_state.get_continuous_play_active():
                # Automatically get new prediction and execute
                try:
                    # End policy execution to clear input lock
                    policy_state.end_policy_execution()
                    # Run policy to get prediction horizon
                    success = run_policy(
                        data_manager,
                        policy,
                        policy_state,
                        visualizer,
                        model_input_order,
                    )
                    if not success:
                        print("⚠️  Failed to run policy")
                        end_policy_play(
                            data_manager,
                            policy_state,
                            visualizer,
                            "Continuous play stopped - prediction failed",
                        )
                        continue

                    # Execute policy
                    success = start_policy_execution(data_manager, policy_state)
                    if not success:
                        print("⚠️  Failed to execute policy")
                        end_policy_play(
                            data_manager,
                            policy_state,
                            visualizer,
                            "Continuous play stopped - execution failed",
                        )
                        continue

                except Exception as e:
                    print(f"✗ Failed to get next policy prediction: {e}")
                    traceback.print_exc()
                    end_policy_play(
                        data_manager,
                        policy_state,
                        visualizer,
                        "Continuous play stopped - prediction failed",
                    )
            else:
                # Execution complete
                print("✓ Policy execution completed")
                end_policy_play(
                    data_manager, policy_state, visualizer, "Policy execution completed"
                )

        # NOTE: Update visualization less frequently to avoid blocking
        # Throttle visualization updates to ~30Hz to prevent overwhelming Viser server
        current_time = time.time()
        if current_time - last_visualization_update >= visualization_update_interval:
            update_visualization(data_manager, policy_state, visualizer)
            last_visualization_update = current_time

        dt_execution = 1.0 / visualizer.get_policy_execution_rate()
        elapsed = time.time() - start_time
        if elapsed < dt_execution:
            time.sleep(dt_execution - elapsed)


def update_visualization(
    data_manager: DataManager,
    policy_state: PolicyState,
    visualizer: RobotVisualizer,
) -> None:
    """Update visualization."""
    # Update actual robot visualization
    current_joint_angles = data_manager.get_current_joint_angles()
    if current_joint_angles is not None:
        joint_config_rad = np.radians(current_joint_angles)
        visualizer.update_robot_pose(joint_config_rad)

    # Get policy state for ghost robot
    prediction_horizon = policy_state.get_prediction_horizon()
    prediction_horizon_length = policy_state.get_prediction_horizon_length()
    ghost_robot_playing = policy_state.get_ghost_robot_playing()
    ghost_action_index = policy_state.get_ghost_action_index()

    # Update ghost robot based on current state
    robot_activity_state = data_manager.get_robot_activity_state()
    if robot_activity_state == RobotActivityState.POLICY_CONTROLLED:
        # During policy execution, make ghost robot show target joint angles
        visualizer.update_ghost_robot_visibility(True)
        target_joint_angles = data_manager.get_target_joint_angles()
        if target_joint_angles is not None:
            joint_config_rad = np.radians(target_joint_angles)
            visualizer.update_ghost_robot_pose(joint_config_rad)
        # Disable buttons during execution
        visualizer.set_start_policy_execution_button_disabled(True)
        visualizer.set_run_policy_button_disabled(True)
        visualizer.set_run_and_start_policy_execution_button_disabled(True)
        # Play/Stop button is enabled during execution so we can stop if needed
        visualizer.set_play_policy_button_disabled(False)

    elif (
        robot_activity_state == RobotActivityState.ENABLED
        and data_manager.get_teleop_active()
    ):
        # During teleoperation, make ghost robot show target joint angles
        visualizer.update_ghost_robot_visibility(True)
        target_joint_angles = data_manager.get_target_joint_angles()
        if target_joint_angles is not None:
            joint_config_rad = np.radians(target_joint_angles)
            visualizer.update_ghost_robot_pose(joint_config_rad)

    elif ghost_robot_playing and prediction_horizon_length > 0:
        # Enable execute policy button
        visualizer.set_start_policy_execution_button_disabled(False)
        # show ghost robot
        visualizer.update_ghost_robot_visibility(True)
        # Update ghost robot with prediction horizon actions (preview mode)
        if ghost_action_index < prediction_horizon_length:
            # Get ghost action from horizon
            if all(joint_name in prediction_horizon for joint_name in JOINT_NAMES):
                ghost_joint_config = np.array(
                    [
                        prediction_horizon[joint_name][ghost_action_index]
                        for joint_name in JOINT_NAMES
                    ]
                )
                visualizer.update_ghost_robot_pose(ghost_joint_config)
            next_index = (ghost_action_index + 1) % prediction_horizon_length
            policy_state.set_ghost_action_index(next_index)
        else:
            policy_state.reset_ghost_action_index()

    else:
        # When not playing, hide the ghost robot
        visualizer.update_ghost_robot_visibility(False)

        # Update button state and policy status when not policy controlled
        robot_enabled = robot_activity_state == RobotActivityState.ENABLED
        has_horizon = prediction_horizon_length > 0

        # Update button enabled state
        visualizer.set_start_policy_execution_button_disabled(
            not (robot_enabled and has_horizon)
        )
        visualizer.set_run_policy_button_disabled(not robot_enabled)
        visualizer.set_run_and_start_policy_execution_button_disabled(not robot_enabled)
        visualizer.set_play_policy_button_disabled(not robot_enabled)

        # Update policy status
        if not has_horizon:
            visualizer.update_policy_status(
                "Ready - Press Right Joystick or 'Run Policy' button to get prediction"
            )
        elif not robot_enabled:
            visualizer.update_policy_status("Robot not enabled")
        else:
            visualizer.update_policy_status(
                f"Ready - {prediction_horizon_length} actions in horizon"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Piper Robot Test with Neuracore Policy - REAL ROBOT CONTROL"
    )
    parser.add_argument(
        "--ip-address",
        type=str,
        default=None,
        help="IP address of Meta Quest device (optional, defaults to None for auto-discovery)",
    )
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
    print("PIPER ROBOT TEST WITH NEURACORE POLICY")
    print("=" * 60)
    print("Thread frequencies:")
    print(f"  🎮 Quest Controller: {CONTROLLER_DATA_RATE} Hz")
    print(f"  🧮 IK Solver:        {IK_SOLVER_RATE} Hz")
    print(f"  🤖 Robot Controller: {ROBOT_RATE} Hz")
    print(f"  📸 Camera Frame:     {CAMERA_FRAME_STREAMING_RATE} Hz")
    print(f"  📊 Joint State:      {JOINT_STATE_STREAMING_RATE} Hz")
    print(f"  🖥️  Visualization:    {VISUALIZATION_RATE} Hz")

    # Connect to Neuracore
    print("\n🔧 Initializing Neuracore...")
    nc.login()
    nc.connect_robot(
        robot_name="AgileX PiPER",
        urdf_path=str(URDF_PATH),
        overwrite=False,
    )

    # Load policy from either train run name or model path
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
        DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS: [GRIPPER_LOGGING_NAME],
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
            device="cuda",
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

    # Initialize policy state
    policy_state = PolicyState()
    policy_state.set_execution_mode(PolicyState.ExecutionMode.TARGETING_TIME)

    # Initialize shared state
    data_manager = DataManager()
    data_manager.set_controller_filter_params(
        CONTROLLER_MIN_CUTOFF,
        CONTROLLER_BETA,
        CONTROLLER_D_CUTOFF,
    )

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

    # Start joint state thread
    print("\n📊 Starting joint state thread...")
    joint_state_thread_obj = threading.Thread(
        target=joint_state_thread, args=(data_manager, robot_controller), daemon=True
    )
    joint_state_thread_obj.start()

    # Initialize Meta Quest reader
    print("\n🎮 Initializing Meta Quest reader...")
    quest_reader = MetaQuestReader(ip_address=args.ip_address, port=5555, run=True)

    # Start data collection thread
    print("\n🎮 Starting quest reader thread...")
    quest_thread = threading.Thread(
        target=quest_reader_thread, args=(data_manager, quest_reader), daemon=True
    )
    quest_thread.start()

    # set initial configuration to current joint angles
    current_joint_angles = data_manager.get_current_joint_angles()
    if current_joint_angles is not None:
        initial_joint_angles = np.radians(current_joint_angles)
    else:
        initial_joint_angles = np.radians(NEUTRAL_JOINT_ANGLES)

    # Create Pink IK solver
    print("\n🔧 Creating Pink IK solver...")
    ik_solver = PinkIKSolver(
        urdf_path=URDF_PATH,
        end_effector_frame=GRIPPER_FRAME_NAME,
        solver_name=SOLVER_NAME,
        position_cost=POSITION_COST,
        orientation_cost=ORIENTATION_COST,
        frame_task_gain=FRAME_TASK_GAIN,
        lm_damping=LM_DAMPING,
        damping_cost=DAMPING_COST,
        solver_damping_value=SOLVER_DAMPING_VALUE,
        integration_time_step=1 / IK_SOLVER_RATE,
        initial_configuration=initial_joint_angles,
        posture_cost_vector=np.array(POSTURE_COST_VECTOR),
    )

    # Start IK solver thread
    print("\n🧮 Starting IK solver thread...")
    ik_thread = threading.Thread(
        target=ik_solver_thread, args=(data_manager, ik_solver), daemon=True
    )
    ik_thread.start()

    # Start camera thread
    print("\n📷 Starting camera thread...")
    camera_thread_obj = threading.Thread(
        target=camera_thread, args=(data_manager,), daemon=True
    )
    camera_thread_obj.start()

    # Set up visualization
    print("\n🖥️  Starting Viser visualization...")
    visualizer = RobotVisualizer(str(URDF_PATH))
    visualizer.add_policy_controls(
        initial_prediction_ratio=PREDICTION_HORIZON_EXECUTION_RATIO,
        initial_policy_rate=POLICY_EXECUTION_RATE,
        initial_robot_rate=ROBOT_RATE,
        initial_execution_mode=PolicyState.ExecutionMode.TARGETING_TIME.value,
    )
    visualizer.add_toggle_robot_enabled_status_button()
    visualizer.add_homing_controls()
    visualizer.add_policy_buttons()

    # Set up button callbacks
    visualizer.set_toggle_robot_enabled_status_callback(
        lambda: toggle_robot_enabled_status(data_manager, robot_controller, visualizer)
    )
    visualizer.set_go_home_callback(lambda: home_robot(data_manager, robot_controller))
    visualizer.set_run_policy_callback(
        lambda: run_policy(
            data_manager, policy, policy_state, visualizer, model_input_order
        )
    )
    visualizer.set_start_policy_execution_callback(
        lambda: start_policy_execution(data_manager, policy_state)
    )
    visualizer.set_run_and_start_policy_execution_callback(
        lambda: run_and_start_policy_execution(
            data_manager, policy, policy_state, visualizer, model_input_order
        )
    )
    visualizer.set_play_policy_callback(
        lambda: play_policy(
            data_manager, policy, policy_state, visualizer, model_input_order
        )
    )
    # Set up execution mode dropdown callback to sync with PolicyState
    visualizer.set_execution_mode_callback(
        lambda: policy_state.set_execution_mode(
            PolicyState.ExecutionMode(visualizer.get_execution_mode())
        )
    )

    # Register Quest reader button callbacks (after visualizer is created)
    quest_reader.on(
        "button_a_pressed",
        lambda: toggle_robot_enabled_status(data_manager, robot_controller, visualizer),
    )
    quest_reader.on(
        "button_b_pressed", lambda: home_robot(data_manager, robot_controller)
    )

    # Start policy execution thread
    print("\n🤖 Starting policy execution thread...")
    policy_execution_thread_obj = threading.Thread(
        target=policy_execution_thread,
        args=(
            policy,
            data_manager,
            policy_state,
            robot_controller,
            visualizer,
            model_input_order,
        ),
        daemon=True,
    )
    policy_execution_thread_obj.start()

    print()
    print("🚀 Starting teleoperation with policy testing...")
    print("🎮 CONTROLS:")
    print("   1. Press BUTTON A or Enable Robot button to enable/disable robot")
    print("   2. You have same control over the robot as in teleoperation.")
    print("      - Hold RIGHT GRIP to activate teleoperation")
    print("      - Move controller - robot follows!")
    print("      - Hold RIGHT TRIGGER to close gripper")
    print("      - Press BUTTON A or Enable Robot button to enable/disable robot")
    print("      - Press BUTTON B or Home Robot button to send robot home")
    print("   3. Click 'Run Policy' button to run policy (without executing)")
    print("   4. Click 'Execute Policy' button to execute prediction horizon")
    print("   5. Click 'Run and Execute Policy' button to run and execute policy")
    print("   6. Click 'Play Policy' button to play policy")
    print("⚠️  Press Ctrl+C to exit")
    print()
    print("🌐 Open browser: http://localhost:8080")

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n👋 Interrupt received - shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        traceback.print_exc()

    # Cleanup
    print("\n🧹 Cleaning up...")

    # Disconnect policy
    policy.disconnect()

    # shutdown threads
    data_manager.request_shutdown()
    data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
    quest_thread.join()
    quest_reader.stop()
    ik_thread.join()
    camera_thread_obj.join()
    robot_controller.cleanup()

    nc.logout()

    print("\n👋 Demo stopped.")
