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
from neuracore_types import DataType, EmbodimentDescription

# Add parent directory to path to import pink_ik_solver and piper_controller
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import (
    CAMERA_FRAME_STREAMING_RATE,
    CAMERA_NAMES,
    CONTROLLER_BETA,
    CONTROLLER_D_CUTOFF,
    CONTROLLER_DATA_RATE,
    CONTROLLER_MIN_CUTOFF,
    DAMPING_COST,
    FRAME_TASK_GAIN,
    GRIPPER_FRAME_NAME,
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
from common.robot_visualizer import RobotVisualizer
from common.threads.ik_solver import ik_solver_thread
from common.threads.joint_state import joint_state_thread
from common.threads.quest_reader import quest_reader_thread
from common.threads.realsense_camera import camera_thread
from meta_quest_teleop.reader import MetaQuestReader

from pink_ik_solver import PinkIKSolver
from piper_controller import PiperController


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
    input_embodiment_description: EmbodimentDescription,
) -> bool:
    """Handle Run Policy button press to capture state and get policy prediction."""
    print("Running policy...")

    if not log_robot_state_for_policy(data_manager, input_embodiment_description):
        print("✗ No data available to run policy")
        return False

    rgb_image = None
    if DataType.RGB_IMAGES in input_embodiment_description:
        rgb_names = embodiment_names_ordered(
            input_embodiment_description[DataType.RGB_IMAGES]
        )
        if rgb_names:
            rgb_image = data_manager.get_rgb_image(rgb_names[0])
    current_joint_angles = data_manager.get_current_joint_angles()

    # Get policy prediction
    try:
        start_time = time.time()
        predictions = policy.predict(timeout=60)
        prediction_horizon = convert_predictions_to_horizon(predictions)
        end_time = time.time()
        
        # Calculate length directly from the new prediction dictionary
        horizon_length = 0
        if prediction_horizon:
            first_key = next(iter(prediction_horizon.keys()))
            horizon_length = len(prediction_horizon[first_key])
            
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
    current_target_deg = np.degrees(current_joint_target_positions_rad)
    joint_differences = np.abs(current_joint_angles - current_target_deg)
    
    if np.any(joint_differences > MAX_SAFETY_THRESHOLD):
        print("⚠️ Cannot execute policy: Robot too far from first predicted action")
        print("   --- DIAGNOSTICS ---")
        print(f"   Current Angles: {[f'{d:.2f}' for d in current_joint_angles]}")
        print(f"   AI Predicted:   {[f'{d:.2f}' for d in current_target_deg]}")
        print(f"   Differences:    {[f'{d:.2f}' for d in joint_differences]}")
        print(f"   Threshold:      {MAX_SAFETY_THRESHOLD}°")
        print("   💡 TIP 1: Did the arm sag? Check if 'Current Angles' are drooping.")
        print("   💡 TIP 2: If the AI naturally predicts large first steps, increase MAX_SAFETY_THRESHOLD in common/configs.py")
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


def end_policy_play(
    data_manager: DataManager,
    policy_state: PolicyState,
    visualizer: RobotVisualizer,
    policy_status_message: str,
) -> None:
    """End continuous play and set robot activity state to ENABLED and update policy status."""
    if policy_state.get_continuous_play_active():
        policy_state.set_continuous_play_active(False)
    
    # Reset ghost robot color to default orange
    visualizer.set_ghost_robot_color((1.0, 0.65, 0.0, 0.25))
    
    visualizer.update_play_policy_button_status(False)

    visualizer.update_play_policy_button_status(False)
    policy_state.end_policy_execution()
    data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
    data_manager.set_teleop_state(False, None, None)
    visualizer.update_policy_status(policy_status_message)

def continuous_prediction_worker(
    data_manager: DataManager,
    policy: nc.policy,
    policy_state: PolicyState,
    visualizer: RobotVisualizer,
    input_embodiment_description: EmbodimentDescription,
    continuous_mode: str = "pipeline",
) -> None:
    """Background thread for continuous execution supporting pipelined and sequential modes."""
    VISUALIZATION_COLORS = [
        (1.0, 0.65, 0.0, 0.25),  # Orange (Default)
        (0.0, 1.0, 0.0, 0.25),   # Green
        (1.0, 0.0, 0.0, 0.25),   # Red
        (0.0, 0.0, 1.0, 0.25),   # Blue
    ]
    color_index = 0

    # 1. Bootstrap the very first prediction to get the robot moving
    print(f"\n🚀 [Worker] Bootstrapping initial trajectory in '{continuous_mode}' mode...")
    success = run_policy(data_manager, policy, policy_state, visualizer, input_embodiment_description)
    if success:
        start_policy_execution(data_manager, policy_state)

    while policy_state.get_continuous_play_active():
        # Failsafe: if there's no active trajectory running yet, wait briefly
        if policy_state.get_locked_prediction_horizon_length() == 0:
            time.sleep(0.01)
            continue

        if continuous_mode == "pipeline":
            print("\n📸 [Pipeline Worker] Robot is moving! Prefetching next prediction in background...")
            # Query the network in parallel while the execution thread is driving the motors
            success = run_policy(data_manager, policy, policy_state, visualizer, input_embodiment_description)
            
            if not success or not policy_state.get_continuous_play_active():
                time.sleep(0.05)
                continue

            # Wait until the current trajectory buffer is running low before swapping
            while policy_state.get_continuous_play_active():
                exec_idx = policy_state.get_execution_action_index()
                total_len = policy_state.get_locked_prediction_horizon_length()
                remaining = total_len - exec_idx
                
                # Hot-swap when 5 or fewer steps are left in the active trajectory
                if remaining <= 5 or total_len == 0:
                    break
                time.sleep(0.01)

        elif continuous_mode == "sequential":
            # Wait until the current trajectory buffer is completely exhausted
            while policy_state.get_continuous_play_active():
                exec_idx = policy_state.get_execution_action_index()
                total_len = policy_state.get_locked_prediction_horizon_length()
                if exec_idx >= total_len or total_len == 0:
                    break
                time.sleep(0.01)

            if not policy_state.get_continuous_play_active():
                break

            print("\n📸 [Sequential Worker] Trajectory finished! Holding position and requesting next prediction...")
            success = run_policy(data_manager, policy, policy_state, visualizer, input_embodiment_description)

            if not success or not policy_state.get_continuous_play_active():
                time.sleep(0.05)
                continue

        if not policy_state.get_continuous_play_active():
            break

        print("🔄 [Worker] Swapping to new trajectory buffer!")
        # Seamlessly clear the lock and flash the new horizon into play
        policy_state.end_policy_execution()
        success = start_policy_execution(data_manager, policy_state)
        
        if success:
            color_index = (color_index + 1) % len(VISUALIZATION_COLORS)
            visualizer.set_ghost_robot_color(VISUALIZATION_COLORS[color_index])
        else:
            print("❌ [Worker] Swap rejected by safety threshold. Retrying immediately...")
            time.sleep(0.01)

def play_policy(
    data_manager: DataManager,
    policy: nc.policy,
    policy_state: PolicyState,
    visualizer: RobotVisualizer,
    input_embodiment_description: EmbodimentDescription,
    continuous_mode: str = "pipeline",
) -> None:
    """Handle Play Policy button press to start/stop continuous policy execution."""
    if not policy_state.get_continuous_play_active():
        # Start continuous play
        print(f"▶️  Play Policy button pressed - Starting {continuous_mode.capitalize()} Mode...")
        policy_state.set_continuous_play_active(True)
        visualizer.update_play_policy_button_status(True)
        
        # Spawn the background worker
        threading.Thread(
            target=continuous_prediction_worker,
            args=(data_manager, policy, policy_state, visualizer, input_embodiment_description, continuous_mode),
            daemon=True
        ).start()
    else:
        # Stop continuous play
        print("⏹️  Stop Policy button pressed - Stopping continuous policy execution...")
        policy_state.set_continuous_play_active(False)
        end_policy_play(
            data_manager, policy_state, visualizer, "Policy execution stopped"
        )
        print("✓ Policy execution stopped and robot enabled")

def policy_execution_thread(
    policy: nc.policy,
    data_manager: DataManager,
    policy_state: PolicyState,
    robot_controller: PiperController,
    visualizer: RobotVisualizer,
    input_embodiment_description: EmbodimentDescription,
    output_gripper_names: list[str] | None,
) -> None:
    """Policy execution thread."""
    dt_execution = 1.0 / POLICY_EXECUTION_RATE
    
    # Define colors for continuous horizon visualization
    VISUALIZATION_COLORS = [
        (1.0, 0.65, 0.0, 0.25),  # Orange (Default)
        (0.0, 1.0, 0.0, 0.25),  # Green
        (1.0, 0.0, 0.0, 0.25),  # Red
        (0.0, 0.0, 1.0, 0.25),  # Blue
    ]
    color_index = 0

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

            # If continuous play is active, only execute up to the chunk limit
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
                        previous_joint_target_positions_deg = (
                            joint_targets_deg_at_index(
                                locked_horizon, execution_index - 1
                            )
                        )
                        if previous_joint_target_positions_deg is None:
                            break
                        joint_errors = np.abs(
                            current_joint_angles - previous_joint_target_positions_deg
                        )
                        if np.any(joint_errors <= MAX_ACTION_ERROR_THRESHOLD):
                            break
                        time.sleep(0.001)

                current_joint_target_positions_deg = joint_targets_deg_at_index(
                    locked_horizon, execution_index
                )
                if current_joint_target_positions_deg is not None:
                    data_manager.set_target_joint_angles(
                        current_joint_target_positions_deg
                    )
                    if robot_controller.is_robot_enabled():
                        robot_controller.set_target_joint_angles(
                            current_joint_target_positions_deg
                        )
                    else:
                        print(
                            f"⚠️  Robot controller not enabled, skipping command at index {execution_index}"
                        )

                gripper_target = gripper_open_at_index(
                    locked_horizon,
                    execution_index,
                    gripper_names=output_gripper_names,
                )
                if gripper_target is not None:
                    robot_controller.set_gripper_open_value(gripper_target)

                # Update execution index
                policy_state.increment_execution_action_index()

                # Update status
                visualizer.update_policy_status(
                    f"Executing policy: {execution_index + 1}/{locked_horizon_length}"
                )
            else:
                # Horizon buffer exhausted
                if not policy_state.get_continuous_play_active():
                    print("✓ Policy execution completed")
                    end_policy_play(
                        data_manager, policy_state, visualizer, "Policy execution completed"
                    )
                else:
                    # Failsafe: If the background thread is running slightly late, 
                    # hold the very last predicted position to maintain motor torque.
                    if all(joint_name in locked_horizon for joint_name in JOINT_NAMES):
                        last_index = locked_horizon_length - 1
                        hold_positions_rad = np.array([
                            locked_horizon[jn][last_index] for jn in JOINT_NAMES
                        ])
                        if robot_controller.is_robot_enabled():
                            robot_controller.set_target_joint_angles(np.degrees(hold_positions_rad))

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

    # Update RGB camera image in Viser GUI (if available)
    rgb_image = data_manager.get_rgb_image(CAMERA_NAMES[0])
    if rgb_image is not None:
        visualizer.update_rgb_image(rgb_image)

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
        # visualizer.set_start_policy_execution_button_disabled(True)
        visualizer.set_run_policy_button_disabled(True)
        # visualizer.set_run_and_start_policy_execution_button_disabled(True)
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
        #visualizer.set_run_and_start_policy_execution_button_disabled(not robot_enabled)
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
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument(
        "--train-run-name",
        type=str,
        default=None,
        help="Name of the training run to load policy from (for cloud training).",
    )

    parser.add_argument(
        "--continuous-mode",
        type=str,
        choices=["pipeline", "sequential"],
        default="pipeline",
        help="Execution mode for Play Policy: 'pipeline' (smooth hot-swapping) or 'sequential' (execute full horizon, then pause to predict next).",
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
    args = parser.parse_args()

    print("=" * 60)
    print("PIPER ROBOT TEST WITH NEURACORE POLICY")
    print("=" * 60)
    print("Thread frequencies:")
    print(f"  🎮 Quest Controller: {CONTROLLER_DATA_RATE} Hz")
    print(f"  🧮 IK Solver:        {IK_SOLVER_RATE} Hz")
    print(f"  🤖 Robot Controller: {ROBOT_RATE} Hz")
    print(f"  📸 Camera Frame:     {CAMERA_FRAME_STREAMING_RATE} Hz")
    print(f"  📊 Joint State:      {JOINT_STATE_STREAMING_RATE} Hz")
    print(f"  🖥️  Visualization:   {VISUALIZATION_RATE} Hz")

    # Connect to Neuracore
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
            data_manager,
            policy,
            policy_state,
            visualizer,
            input_embodiment_description,
        )
    )
    visualizer.set_start_policy_execution_callback(
        lambda: start_policy_execution(data_manager, policy_state)
    )

    visualizer.set_play_policy_callback(
        lambda: play_policy(
            data_manager,
            policy,
            policy_state,
            visualizer,
            input_embodiment_description,
            args.continuous_mode,
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
            input_embodiment_description,
            output_gripper_names,
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
    print("   3. Click 'Run Policy' (Preview) to generate and visualize a prediction horizon")
    print("   4. Click 'Execute Policy' to run the currently previewed horizon")
    print("   5. Click 'Play Policy' (Receding Horizon) to constantly predict and execute the first action")
    # print("   6. Click 'Play Policy' button to play policy")
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
