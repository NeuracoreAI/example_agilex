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

import numpy as np

# Add parent directory to path to import pink_ik_solver and piper_controller
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import (
    CAMERA_FRAME_STREAMING_RATE,
    CONTROLLER_BETA,
    CONTROLLER_D_CUTOFF,
    CONTROLLER_DATA_RATE,
    CONTROLLER_MIN_CUTOFF,
    DAMPING_COST,
    FRAME_TASK_GAIN,
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
from common.threads.camera import camera_thread
from common.threads.ik_solver import ik_solver_thread
from common.threads.joint_state import joint_state_thread
from common.threads.quest_reader import quest_reader_thread
from meta_quest_teleop.reader import MetaQuestReader

from pink_ik_solver import PinkIKSolver
from piper_controller import PiperController


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
        ok = robot_controller.move_to_home()
        if not ok:
            print("✗ Failed to initiate home move")
            # Revert to ENABLED on failure
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
    else:
        print("⚠️  Button B pressed but robot is not enabled")


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
visualizer.add_toggle_robot_enabled_status_button()
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


# Set up button callbacks
def toggle_robot_enabled_status() -> None:
    """Toggle robot enabled/disabled state and update GUI button label."""
    robot_activity_state = data_manager.get_robot_activity_state()
    if robot_activity_state == RobotActivityState.ENABLED:
        # Disable robot
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        robot_controller.graceful_stop()
        # Reset teleop state when disabling robot
        data_manager.set_teleop_state(False, None, None)
        visualizer.update_toggle_robot_enabled_status(False)
        print("✓ 🔴 Robot disabled")
    elif robot_activity_state == RobotActivityState.DISABLED:
        if robot_controller.resume_robot():
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
            visualizer.update_toggle_robot_enabled_status(True)
            print("✓ 🟢 Robot enabled")
        else:
            print("✗ Failed to enable robot")


def on_go_home() -> None:
    """Handle Button B press to move robot to home position."""
    robot_activity_state = data_manager.get_robot_activity_state()
    if robot_activity_state == RobotActivityState.ENABLED:
        print("🏠 GUI: Moving to home position...")
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
        data_manager.set_teleop_state(False, None, None)
        ok = robot_controller.move_to_home()
        if not ok:
            print("✗ Failed to initiate home move")
            # Revert to ENABLED on failure
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
    else:
        print("⚠️  Cannot home: robot not enabled")


visualizer.set_toggle_robot_enabled_status_callback(toggle_robot_enabled_status)
visualizer.set_go_home_callback(on_go_home)

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
        # Get updated robot_activity_state and joint states (may have changed if homing completed)
        robot_activity_state = data_manager.get_robot_activity_state()
        current_joint_angles = data_manager.get_current_joint_angles()
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

        # Update main robot visualization from CURRENT joint angles (DataManager uses degrees, Viser expects radians)
        if current_joint_angles is not None:
            current_joint_rad = np.radians(current_joint_angles)
            visualizer.update_robot_pose(current_joint_rad)
            visualizer.update_joint_angles_display(current_joint_rad)

        # Update ghost robot to show TARGET joint angles when available (also in radians)
        if (
            target_joint_angles is not None
            and robot_activity_state == RobotActivityState.ENABLED
        ):
            target_joint_rad = np.radians(target_joint_angles)
            visualizer.update_ghost_robot_visibility(True)
            visualizer.update_ghost_robot_pose(target_joint_rad)
        else:
            # Hide ghost robot when no valid target is available or robot not enabled
            visualizer.update_ghost_robot_visibility(False)

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
