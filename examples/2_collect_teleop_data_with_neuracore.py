#!/usr/bin/env python3
"""Piper Robot Teleoperation with Meta Quest Controller and Neuracore data collection.

This demo uses Pink IK control with Meta Quest controller input to control the Piper robot and
logs data to Neuracore.
"""


import argparse
import multiprocessing
import sys
import threading
import time
import traceback
from pathlib import Path

import neuracore as nc
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
    GRIPPER_LOGGING_NAME,
    IK_SOLVER_RATE,
    JOINT_NAMES,
    JOINT_STATE_STREAMING_RATE,
    LM_DAMPING,
    NEUTRAL_JOINT_ANGLES,
    ORIENTATION_COST,
    POSITION_COST,
    POSTURE_COST_VECTOR,
    ROBOT_RATE,
    SOLVER_DAMPING_VALUE,
    SOLVER_NAME,
    URDF_PATH,
)
from common.data_manager import DataManager, RobotActivityState
from common.threads.camera import camera_thread
from common.threads.ik_solver import ik_solver_thread
from common.threads.joint_state import joint_state_thread
from common.threads.quest_reader import quest_reader_thread
from meta_quest_teleop.reader import MetaQuestReader

from pink_ik_solver import PinkIKSolver
from piper_controller import PiperController


def log_to_neuracore_on_change_callback(
    name: str, value: float, timestamp: float
) -> None:
    """Log data to queue on change callback."""
    # Call appropriate Neuracore logging function
    try:
        if name == "log_joint_positions":
            data_value = np.radians(value)
            data_dict = {
                joint_name: angle for joint_name, angle in zip(JOINT_NAMES, data_value)
            }
            nc.log_joint_positions(data_dict, timestamp=timestamp)
        elif name == "log_joint_target_positions":
            data_value = np.radians(value)
            data_dict = {
                joint_name: angle for joint_name, angle in zip(JOINT_NAMES, data_value)
            }
            nc.log_joint_target_positions(data_dict, timestamp=timestamp)
        elif name == "log_parallel_gripper_open_amounts":
            data_dict = {GRIPPER_LOGGING_NAME: value}
            nc.log_parallel_gripper_open_amounts(data_dict, timestamp=timestamp)
        elif name == "log_parallel_gripper_target_open_amounts":
            data_dict = {GRIPPER_LOGGING_NAME: value}
            nc.log_parallel_gripper_target_open_amounts(data_dict, timestamp=timestamp)
        elif name == "log_rgb":
            camera_name = "rgb"
            image_array = value
            nc.log_rgb(camera_name, image_array, timestamp=timestamp)
        else:
            print(f"\n⚠️  Unknown logging function: {name}")
    except Exception as e:
        print(f"\n⚠️  Failed to call {name} to Neuracore: {e}")


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


def on_button_rj_pressed() -> None:
    """Handle Button Right Joystick press to toggle data recording state."""
    if not nc.is_recording():
        # Start recording
        try:
            nc.start_recording()
            print("✓ 🔴 Recording started (Button RJ)")
        except Exception as e:
            print(f"✗ Failed to start recording: {e}")
    else:
        # Stop recording
        try:
            nc.stop_recording()
            print("✓ ⏹️ Recording stopped (Button RJ)")
        except Exception as e:
            print(f"✗ Failed to stop recording: {e}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        description="Piper Robot Teleoperation with Neuracore Data Collection - REAL ROBOT CONTROL"
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
        help="Name for the dataset (optional, defaults to auto-generated timestamp-based name)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PIPER ROBOT TELEOPERATION - REAL ROBOT CONTROL")
    print("=" * 60)
    print("Thread frequencies:")
    print(f"  🎮 Quest Controller: {CONTROLLER_DATA_RATE} Hz")
    print(f"  🧮 IK Solver:        {IK_SOLVER_RATE} Hz")
    print(f"  🤖 Robot Controller: {ROBOT_RATE} Hz")
    print(f"  📸 Camera Frame:     {CAMERA_FRAME_STREAMING_RATE} Hz")
    print(f"  📊 Joint State:      {JOINT_STATE_STREAMING_RATE} Hz")

    # Connect to Neuracore
    print("\n🔧 Initializing Neuracore...")
    nc.login()
    nc.connect_robot(
        robot_name="AgileX PiPER",
        urdf_path=str(URDF_PATH),
        overwrite=False,
    )

    # Create dataset
    dataset_name = (
        args.dataset_name or f"piper-teleop-data-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    print(f"\n🔧 Creating dataset {dataset_name}...")
    nc.create_dataset(
        name=dataset_name,
        description="Teleop data collection for Piper robot",
    )

    # Initialize shared state
    data_manager = DataManager()
    data_manager.set_on_change_callback(log_to_neuracore_on_change_callback)
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
    quest_reader = MetaQuestReader(
        ip_address=args.ip_address,
        port=5555,
        run=True,
    )

    # Register button callbacks (after state and robot_controller are initialized)
    quest_reader.on("button_a_pressed", on_button_a_pressed)
    quest_reader.on("button_b_pressed", on_button_b_pressed)
    quest_reader.on("button_rj_pressed", on_button_rj_pressed)

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

    # Start camera thread (if RealSense is available)
    print("\n📷 Starting camera thread...")
    camera_thread_obj = threading.Thread(
        target=camera_thread, args=(data_manager,), daemon=True
    )
    camera_thread_obj.start()

    print()
    print("🚀 Starting teleoperation with REAL ROBOT CONTROL...")
    print("🎮 CONTROLS:")
    print("   1. Press BUTTON A to enable/disable robot")
    print("   2. Hold RIGHT GRIP to activate teleoperation")
    print("   3. Move controller - robot follows!")
    print("   4. Hold RIGHT TRIGGER to close gripper")
    print("   5. Press BUTTON B to send robot home")
    print("   6. Press RIGHT JOYSTICK to start/stop data recording")
    print("   7. Release grip to stop")
    print("⚠️  Press Ctrl+C to exit")
    print()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Interrupt received - shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback

        traceback.print_exc()

    # Cleanup
    print("\n🧹 Cleaning up...")

    # Cancel recording if active
    if nc.is_recording():
        try:
            print("⚠️  Cancelling active recording...")
            nc.cancel_recording()
            print("✓ Recording cancelled")
        except Exception as e:
            print(f"⚠️  Error cancelling recording: {e}")

    # shutdown threads
    nc.logout()
    data_manager.request_shutdown()
    data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
    quest_thread.join()
    quest_reader.stop()
    ik_thread.join()
    camera_thread_obj.join()
    robot_controller.cleanup()

    print("\n👋 Demo stopped.")
