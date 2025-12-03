#!/usr/bin/env python3
"""Piper Robot Teleoperation with Meta Quest Controller - REAL ROBOT CONTROL.

This demo uses Pink IK control with Meta Quest controller input to control the REAL robot.
- REAL ROBOT CONTROL - sends commands to physical robot!
- Uses right hand controller grip as dead man's button
- Uses ROS pointer frame for natural pointing control
- Applies relative transformations accounting for different coordinate frames
"""

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

# Add parent directory to path to import pink_ik_solver and piper_controller
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add meta_quest_teleop to path
sys.path.insert(0, str(Path(__file__).parent.parent / "meta_quest_teleop"))


from common.configs import (
    CAMERA_FRAME_STREAMING_RATE,
    CONTROLLER_BETA,
    CONTROLLER_D_CUTOFF,
    CONTROLLER_DATA_RATE,
    CONTROLLER_MIN_CUTOFF,
    DAMPING_COST,
    FRAME_TASK_GAIN,
    GRIP_THRESHOLD,
    GRIPPER_FRAME_NAME,
    IK_SOLVER_RATE,
    JOINT_STATE_STREAMING_RATE,
    LM_DAMPING,
    NEUTRAL_JOINT_ANGLES,
    ORIENTATION_COST,
    POSITION_COST,
    POSTURE_COST_VECTOR,
    ROBOT_RATE,
    ROTATION_SCALE,
    SOLVER_DAMPING_VALUE,
    SOLVER_NAME,
    TRANSLATION_SCALE,
    URDF_PATH,
    VISUALIZATION_RATE,
)
from common.data_manager import DataManager, RobotActivityState
from common.robot_visualizer import RobotVisualizer
from common.utils import scale_and_add_delta_transform

from meta_quest_teleop.reader import MetaQuestReader
from pink_ik_solver import PinkIKSolver
from piper_controller import PiperController


def quest_reader_thread(
    data_manager: DataManager, quest_reader: MetaQuestReader
) -> None:
    """Quest reader thread - reads quest controller data.

    This thread runs at high frequency to ensure responsive controller input.
    Handles:
    - Reading Meta Quest controller data
    - Processing grip button (dead man's switch)
    - Managing teleop activation/deactivation
    - Capturing initial poses when teleop activates

    Args:
        data_manager: DataManager object for thread-safe communication
        quest_reader: MetaQuestReader instance
    """
    print("🎮 Quest Controller thread started")

    dt: float = 1.0 / CONTROLLER_DATA_RATE
    prev_grip_active: bool = False

    try:
        while not data_manager.is_shutdown_requested():
            iteration_start = time.time()

            # Get controller data
            grip_value = quest_reader.get_grip_value("right")
            trigger_value = quest_reader.get_trigger_value("right")
            controller_transform = quest_reader.get_hand_controller_transform_ros(
                hand="right"
            )

            # Update shared state with controller data
            data_manager.set_controller_data(
                controller_transform, grip_value, trigger_value
            )

            # Grip button logic (dead man's switch)
            robot_activity_state = data_manager.get_robot_activity_state()
            # Teleop can only be activated if robot is ENABLED (not HOMING or DISABLED)
            grip_active = (
                grip_value >= GRIP_THRESHOLD
                and robot_activity_state == RobotActivityState.ENABLED
            )

            # Rising edge - grip just pressed AND robot is enabled
            if (
                grip_active
                and not prev_grip_active
                and controller_transform is not None
            ):
                # capture initial poses
                controller_initial_transform = controller_transform.copy()
                # Capture initial robot end-effector pose
                robot_initial_transform = data_manager.get_current_end_effector_pose()
                # Start teleop control
                data_manager.set_teleop_state(
                    True, controller_initial_transform, robot_initial_transform
                )
                print("✓ Teleop control activated")
                print(
                    f"  Controller initial position: {controller_initial_transform[:3, 3]}"
                )
                if robot_initial_transform is not None:
                    print(f"  Robot initial position: {robot_initial_transform[:3, 3]}")
                else:
                    print("  Robot initial position: None")

            # Falling edge - grip just released OR robot disabled
            elif not grip_active and prev_grip_active:
                # Stop teleop control
                data_manager.set_teleop_state(False, None, None)
                print("✗ Teleop control deactivated")

            prev_grip_active = grip_active

            # Sleep to maintain loop rate (check shutdown more frequently)
            elapsed = time.time() - iteration_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"❌ Quest reader thread error: {e}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        # Ensure clean exit - deactivate teleop
        data_manager.set_teleop_state(False, None, None)
        print("🎮 Quest Controller thread stopped")


def ik_solver_thread(data_manager: DataManager, ik_solver: PinkIKSolver) -> None:
    """IK solver thread - solves IK and updates state.

    This thread runs at medium frequency to solve IK.
    Handles:
    - Computing target transforms from controller deltas
    - Solving inverse kinematics
    - Updating state with IK solution

    Args:
        data_manager: DataManager - object for thread-safe communication
        ik_solver: PinkIKSolver - instance

    Raises:
        Exception: If an error occurs during the thread execution
    """
    print("🧮 IK solver thread started")

    dt: float = 1.0 / IK_SOLVER_RATE

    try:
        while not data_manager.is_shutdown_requested():
            iteration_start: float = time.time()

            # get current robot joint angles from state
            current_joint_angles = data_manager.get_current_joint_angles()
            current_ik_joint_angles = np.degrees(ik_solver.get_current_configuration())

            # to avoid control hiccups, we only update the IK solver if the current joint angles are close to the IK joint angles
            DIVERGENCE_TOLERANCE = 0.1
            if (
                current_joint_angles is not None
                and current_ik_joint_angles is not None
                and np.all(
                    np.abs(current_joint_angles - current_ik_joint_angles)
                    <= DIVERGENCE_TOLERANCE
                )
            ):
                # NOTE: we use no task update as we are only informing the IK about the actual joint angles
                # to avoid diversion over time between actual joint angles and IK joint angles
                # but we still want the EE task objective to still follow the target pose
                # and not to change to follow this joint configuration (hence no task update)
                ik_solver.set_configuration_no_task_update(
                    np.radians(current_joint_angles)
                )
            # get current end effector pose from IK solver and set in state
            ik_ee_pose = ik_solver.get_current_end_effector_pose()
            data_manager.set_current_end_effector_pose(ik_ee_pose)

            # Get current state
            controller_transform, _, _ = data_manager.get_controller_data()
            teleop_active = data_manager.get_teleop_active()

            if teleop_active and controller_transform is not None:
                # get initial robot and controller transforms from state
                controller_initial, robot_initial = (
                    data_manager.get_initial_robot_controller_transforms()
                )
                if controller_initial is None or robot_initial is None:
                    # Skip if initial transforms not set
                    elapsed = time.time() - iteration_start
                    sleep_time = dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue

                # Calculate delta transform in controller space as translation and rotation matrix
                delta_position = controller_transform[:3, 3] - controller_initial[:3, 3]
                delta_orientation = (
                    controller_transform[:3, :3] @ controller_initial[:3, :3].T
                )

                T_robot_target = scale_and_add_delta_transform(
                    delta_position,
                    delta_orientation,
                    TRANSLATION_SCALE,
                    ROTATION_SCALE,
                    robot_initial,
                )

                ik_solver.set_target_pose(T_robot_target[:3, 3], T_robot_target[:3, :3])
                data_manager.set_target_pose(T_robot_target)

                # start IK solving
                success = ik_solver.solve_ik()

                if success:
                    # Get joint configuration (IK solution)
                    joint_config = np.degrees(ik_solver.get_current_configuration())

                    # Update timing
                    stats = ik_solver.get_statistics()
                    solve_time_ms = stats["last_solve_time_ms"]

                    # Update shared state with IK solution
                    data_manager.set_target_joint_angles(joint_config)

                    # Update timing for visualization
                    data_manager.set_ik_solve_time_ms(solve_time_ms)
                    data_manager.set_ik_success(success)

                else:
                    # IK failed
                    data_manager.set_ik_solve_time_ms(0.0)
                    data_manager.set_ik_success(False)

            else:  # robot is HOMING or DISABLED
                # use current joint angles we got from state
                if current_joint_angles is not None:
                    ik_solver.set_configuration(np.radians(current_joint_angles))

                    # Update target transform based on current end effector pose
                    current_end_effector_pose = (
                        data_manager.get_current_end_effector_pose()
                    )
                    data_manager.set_target_pose(current_end_effector_pose)

                    # Update shared state for visualization
                    data_manager.set_target_joint_angles(current_joint_angles)
                    data_manager.set_ik_solve_time_ms(0.0)
                    data_manager.set_ik_success(True)

            # Sleep to maintain loop rate
            elapsed = time.time() - iteration_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"❌ IK solver thread error: {e}")
        import traceback

        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        print("🧮 IK solver thread stopped")


def joint_state_thread(
    data_manager: DataManager, robot_controller: PiperController
) -> None:
    """Joint state thread.

    This thread reads joint state and end effector pose
    from the robot controller and updates the state object.
    It checks if the robot is homing and changes status to ENABLED once it has completed.
    It also sends the target joint angles to the robot controller.
    """
    print("🔧 Joint state thread started")

    dt: float = 1.0 / JOINT_STATE_STREAMING_RATE
    try:
        while not data_manager.is_shutdown_requested():
            iteration_start: float = time.time()

            # get current joint angles from robot controller and update shared state
            current_joint_angles = robot_controller.get_current_joint_angles()
            if current_joint_angles is not None:
                # set current joint angles in state
                data_manager.set_current_joint_angles(current_joint_angles)

            target_joint_angles = data_manager.get_target_joint_angles()
            _, _, trigger_value = data_manager.get_controller_data()

            # Check if robot is homing and sync IK solver if it has completed
            if data_manager.get_robot_activity_state() == RobotActivityState.HOMING:
                if current_joint_angles is not None:
                    # Check if robot has reached home (within 2 degrees per joint)
                    joint_errors = np.abs(
                        current_joint_angles - robot_controller.HOME_JOINT_ANGLES
                    )
                    MAX_ERROR_TOLERANCE = 2.0  # 2 degrees tolerance
                    if np.all(joint_errors < MAX_ERROR_TOLERANCE):
                        # Transition robot back to ENABLED state
                        data_manager.set_robot_activity_state(
                            RobotActivityState.ENABLED
                        )
                        print("✓ Robot reached home position and is re-enabled")

            elif data_manager.get_robot_activity_state() == RobotActivityState.ENABLED:
                # get target joint angles from state
                if target_joint_angles is not None and trigger_value is not None:
                    # set target joint angles
                    robot_controller.set_target_joint_angles(target_joint_angles)
                    # set gripper command based on trigger value (Invert mapping)
                    gripper_open_value = 1.0 - trigger_value
                    robot_controller.set_gripper_open_value(gripper_open_value)

            # Sleep to maintain streaming rate
            elapsed = time.time() - iteration_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"❌ Joint state thread error: {e}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        print("🔧 Joint state thread stopped")


def camera_thread(data_manager: DataManager) -> None:
    """Camera thread - captures RGB images from RealSense and logs to NeuraCore.

    This thread runs at camera frame rate to capture and log RGB images.
    Handles:
    - Initializing RealSense pipeline
    - Capturing RGB frames
    - Logging images to NeuraCore

    Args:
        data_manager: DataManager object for thread-safe shutdown checking
    """
    print("📷 Camera thread started")

    dt: float = 1.0 / CAMERA_FRAME_STREAMING_RATE
    pipeline: rs.pipeline | None = None

    try:
        # Configure RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()

        # Configure RGB stream (640x480 at CAMERA_FRAME_STREAMING_RATE fps)
        config.enable_stream(
            rs.stream.color,
            640,
            480,
            rs.format.rgb8,
            int(CAMERA_FRAME_STREAMING_RATE),
        )

        # Start streaming
        print(f"  Starting RealSense pipeline at {CAMERA_FRAME_STREAMING_RATE} Hz...")
        pipeline.start(config)

        print("  ✓ RealSense pipeline started successfully")

        while not data_manager.is_shutdown_requested():
            iteration_start = time.time()

            try:
                # Wait for frames
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                if color_frame:
                    # Convert to numpy array (RGB format)
                    color_image = np.asanyarray(color_frame.get_data())
                    # Rotate image 270 degrees (frame is rotated 90 degrees counter-clockwise)
                    color_image = np.rot90(color_image, k=3)

                    # visualise the image
                    try:
                        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Camera", color_image)
                        cv2.waitKey(1)
                    except Exception as e:
                        print(f"⚠️  Failed to visualise image: {e}")

            except RuntimeError as e:
                # Handle RealSense timeout or other runtime errors
                if "timeout" not in str(e).lower():
                    print(f"⚠️  RealSense error: {e}")

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
        # Cleanup pipeline
        if pipeline is not None:
            try:
                pipeline.stop()
                print("  ✓ RealSense pipeline stopped")
            except Exception as e:
                print(f"⚠️  Error stopping pipeline: {e}")
        cv2.destroyAllWindows()
        print("📷 Camera thread stopped")


def on_button_a_pressed() -> None:
    """Handle Button A press to toggle robot enable/disable state."""
    robot_activity_state = data_manager.get_robot_activity_state()
    if robot_activity_state == RobotActivityState.ENABLED:
        # Disable robot
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        robot_controller.graceful_stop()
        # Reset teleop state when disabling robot
        data_manager.set_teleop_state(False, None, None)
        print("✓ 🔴 Robot disabled (Button A)")
    elif robot_activity_state == RobotActivityState.DISABLED:
        if robot_controller.resume_robot():
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
            print("✓ 🟢 Robot enabled (Button A)")
        else:
            print("✗ Failed to enable robot")


def on_button_b_pressed() -> None:
    """Handle Button B press to move robot to home position."""
    robot_activity_state = data_manager.get_robot_activity_state()
    if robot_activity_state == RobotActivityState.ENABLED:
        print("🏠 Button B pressed - Moving to home position...")
        # Set state to HOMING to prevent IK thread from sending robot commands
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
        # Disable teleop during homing
        data_manager.set_teleop_state(False, None, None)
        visualizer.update_homing_status("Homing: In progress (Button B)")
        ok = robot_controller.move_to_home()
        if not ok:
            print("✗ Failed to initiate home move")
            visualizer.update_homing_status("Homing: Failed")
            # Revert to ENABLED on failure
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
        else:
            visualizer.update_homing_status("Homing: In progress (Button B)")
    else:
        print("⚠️  Button B pressed but robot is not enabled")
        visualizer.update_homing_status("Homing: Robot not enabled")


parser = argparse.ArgumentParser(
    description="Piper Robot Teleoperation - REAL ROBOT CONTROL"
)
parser.add_argument(
    "--ip-address",
    type=str,
    default=None,
    help="IP address of Meta Quest device (optional, defaults to None for auto-discovery)",
)
parser.add_argument(
    "--dataset-name",
    type=str,
    default=None,
    help="Optional name for the saved teleop data file",
)
args = parser.parse_args()

print("=" * 60)
print("PIPER ROBOT TELEOPERATION - REAL ROBOT CONTROL")
print("=" * 60)
print("Thread frequencies:")
print(f"  🎮 Quest Reader:     {CONTROLLER_DATA_RATE} Hz")
print(f"  🧮 IK Solver:        {IK_SOLVER_RATE} Hz")
print(f"  🖥️ Visualization:    {VISUALIZATION_RATE} Hz (running in the main thread)")
print(f"  🤖 Robot Controller: {ROBOT_RATE} Hz")
print(f"  📊 Joint State:      {JOINT_STATE_STREAMING_RATE} Hz")
print(f"  📷 Camera:           {CAMERA_FRAME_STREAMING_RATE} Hz")


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

# Register button callbacks (after state and robot_controller are initialized)
quest_reader.on("button_a_pressed", on_button_a_pressed)
quest_reader.on("button_b_pressed", on_button_b_pressed)

# Start quest reader thread
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

# Start camera thread (if RealSense is available)
print("\n📷 Starting camera thread...")
camera_thread_obj = threading.Thread(
    target=camera_thread, args=(data_manager,), daemon=True
)
camera_thread_obj.start()


# Set up visualizer
print("\n🖥️  Starting visualization...")
visualizer = RobotVisualizer(urdf_path=URDF_PATH)
visualizer.add_basic_controls()
visualizer.add_teleop_controls()
visualizer.add_gripper_status_controls()
visualizer.add_homing_controls()
visualizer.add_robot_control_buttons()
visualizer.add_controller_filter_controls(
    initial_min_cutoff=CONTROLLER_MIN_CUTOFF,
    initial_beta=CONTROLLER_BETA,
    initial_d_cutoff=CONTROLLER_D_CUTOFF,
)
visualizer.add_scaling_controls(
    initial_translation_scale=TRANSLATION_SCALE,
    initial_rotation_scale=ROTATION_SCALE,
)
visualizer.add_pink_parameter_controls(
    position_cost=POSITION_COST,
    orientation_cost=ORIENTATION_COST,
    frame_task_gain=FRAME_TASK_GAIN,
    lm_damping=LM_DAMPING,
    damping_cost=DAMPING_COST,
    solver_damping_value=SOLVER_DAMPING_VALUE,
    posture_cost_vector=POSTURE_COST_VECTOR,
)
visualizer.add_controller_visualization()
visualizer.add_target_frame_visualization()

print()
print("🚀 Starting teleoperation with REAL ROBOT CONTROL...")
print("🎮 CONTROLS:")
print("   1. Press BUTTON A to enable/disable robot (or use GUI)")
print("   2. Hold RIGHT GRIP to activate teleoperation")
print("   3. Move controller - robot follows!")
print("   4. Hold RIGHT TRIGGER to close gripper")
print("   5. Press BUTTON B to send robot home (or use GUI)")
print("   6. Release grip to stop")
print("   7. Use 'Emergency Stop' in GUI if needed")
print("⚠️  Press Ctrl+C to exit")
print()

# Visualization loop variables
dt: float = 1.0 / VISUALIZATION_RATE

try:
    while True:
        iteration_start: float = time.time()

        # Handle GUI buttons (robot control)
        if visualizer.is_enable_robot_pressed():
            visualizer.reset_enable_robot_button()
            if robot_controller.resume_robot():
                data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
                print("✓ Robot enabled")
            else:
                print("✗ Failed to enable robot")

        # Handle disable robot button
        if visualizer.is_disable_robot_pressed():
            visualizer.reset_disable_robot_button()
            data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
            robot_controller.graceful_stop()
            # Reset teleop state when disabling robot
            data_manager.set_teleop_state(False, None, None)
            print("✓ Robot disabled")

        # Handle emergency stop button
        if visualizer.is_emergency_stop_pressed():
            visualizer.reset_emergency_stop_button()
            data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
            robot_controller.emergency_stop()
            # Reset teleop state on emergency stop
            data_manager.set_teleop_state(False, None, None)
            print("🚨 Emergency stop activated!")

        # Handle homing button
        if visualizer.is_go_home_pressed():
            visualizer.reset_go_home_button()
            robot_activity_state = data_manager.get_robot_activity_state()
            if robot_activity_state == RobotActivityState.ENABLED:
                print("🏠 GUI: Moving to home position...")
                data_manager.set_robot_activity_state(RobotActivityState.HOMING)
                data_manager.set_teleop_state(False, None, None)
                visualizer.update_homing_status("Homing: In progress (GUI)")
                ok = robot_controller.move_to_home()
                if not ok:
                    print("✗ Failed to initiate home move")
                    visualizer.update_homing_status("Homing: Failed")
                    # Revert to ENABLED on failure
                    data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
                else:
                    visualizer.update_homing_status("Homing: In progress")
            else:
                print("⚠️  Cannot home: robot not enabled")
                visualizer.update_homing_status("Homing: Robot not enabled")

        # Update filter parameters
        min_cutoff, beta, d_cutoff = visualizer.get_controller_filter_params()
        data_manager.set_controller_filter_params(min_cutoff, beta, d_cutoff)

        # Update scaling factors (module-level variables used by IK thread)
        TRANSLATION_SCALE = visualizer.get_translation_scale()
        ROTATION_SCALE = visualizer.get_rotation_scale()

        # Update Pink parameters (GUI controls)
        pink_params = visualizer.get_pink_parameters()
        ik_solver.update_task_parameters(**pink_params)

        # Get data from shared state
        controller_transform, grip_value, trigger_value = (
            data_manager.get_controller_data()
        )
        teleop_active = data_manager.get_teleop_active()
        # Get updated robot_activity_state (may have changed if homing completed)
        robot_activity_state = data_manager.get_robot_activity_state()
        target_joint_angles = data_manager.get_target_joint_angles()
        solve_time_ms = data_manager.get_ik_solve_time_ms()
        ik_success = data_manager.get_ik_success()
        target_pose = data_manager.get_target_pose()

        # Update GUI displays
        visualizer.set_grip_value(grip_value)
        visualizer.set_trigger_value(trigger_value)

        # Update timing display
        visualizer.update_timing(solve_time_ms)

        # Update controller visualization
        visualizer.update_controller_visualization(controller_transform)
        if controller_transform is not None:
            visualizer.update_controller_status_display(
                controller_transform[:3, 3], connected=True
            )
        else:
            visualizer.update_controller_status_display(None, connected=False)

        # Update teleop status display
        visualizer.update_teleop_status(teleop_active)

        # Update target/goal visualization
        visualizer.update_target_visualization(target_pose)

        # Update robot visualization
        if target_joint_angles is not None:
            visualizer.update_robot_pose(target_joint_angles)
            visualizer.update_joint_angles_display(target_joint_angles)

        # Update robot status - simple Enabled/Homing/Disabled
        if robot_activity_state == RobotActivityState.ENABLED:
            visualizer.update_robot_status("Robot Status: Enabled")
        elif robot_activity_state == RobotActivityState.HOMING:
            visualizer.update_robot_status("Robot Status: Homing")
        else:  # DISABLED
            visualizer.update_robot_status("Robot Status: Disabled")

        # Update gripper status
        visualizer.update_gripper_status(
            trigger_value,
            robot_enabled=(robot_activity_state == RobotActivityState.ENABLED),
        )

        # Sleep to maintain visualization rate
        elapsed = time.time() - iteration_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n\n👋 Interrupt received - shutting down gracefully...")
except Exception as e:
    print(f"\n❌ Demo error: {e}")
    import traceback

    traceback.print_exc()

# Cleanup (outside try/except so it always runs)
print("\n🧹 Cleaning up...")

data_manager.request_shutdown()
data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
quest_thread.join()
quest_reader.stop()
ik_thread.join()
joint_state_thread_obj.join()
robot_controller.cleanup()
visualizer.stop()

print("\n👋 Demo stopped.")
