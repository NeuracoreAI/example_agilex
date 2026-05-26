#!/usr/bin/env python3
"""Piper Robot Teleoperation with Meta Quest Controller and Neuracore data collection.

This demo uses Pink IK control with Meta Quest controller input to control the Piper robot and
logs data to Neuracore.
"""


import argparse
import multiprocessing
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

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
    IK_SOLVER_RATE,
    JOINT_STATE_STREAMING_RATE,
    LM_DAMPING,
    META_QUEST_AXIS_MASK,
    NEUTRAL_END_EFFECTOR_POSE,
    NEUTRAL_JOINT_ANGLES,
    ORIENTATION_COST,
    POSITION_COST,
    POSTURE_COST_VECTOR,
    ROBOT_RATE,
    ROTATION_SCALE,
    SLOW_ROTATION_SCALE,
    SLOW_TRANSLATION_SCALE,
    SOLVER_DAMPING_VALUE,
    SOLVER_NAME,
    TRANSLATION_SCALE,
    URDF_PATH,
    WRIST_JOINT_BUTTON_STEP_DEGREES,
)
from common.data_manager import DataManager, RobotActivityState
from common.threads.ik_solver import ik_solver_thread
from common.threads.joint_state import joint_state_thread
from common.threads.quest_reader import quest_reader_thread
from common.threads.realsense_camera import camera_thread as realsense_camera_thread
from meta_quest_teleop.reader import MetaQuestReader

from pink_ik_solver import PinkIKSolver
from piper_controller import PiperController


def play_audio_feedback(action: str) -> None:
    """Play distinct synthesized tones asynchronously using SoX."""
    # Start = High pitch (880 Hz), Stop = Low pitch (440 Hz)
    freq = "880" if action == "start" else "440"
    try:
        # -q: quiet, -n: no input file, synth: generate audio, 0.3: duration, sine: wave type
        subprocess.Popen(
            ["play", "-q", "-n", "synth", "0.3", "sine", freq], 
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"⚠️ Failed to play tone (is SoX installed?): {e}")

def log_to_neuracore_on_change_callback(
    name: str, payload: dict[str, Any], timestamp: float
) -> None:
    """Log data to queue on change callback."""
    # Call appropriate Neuracore logging function
    try:
        #print(f"\n📤 Logging {name} to Neuracore with timestamp {timestamp:.3f}...")
        if name == "log_joint_positions":
            nc.log_joint_positions(payload, timestamp=timestamp)
        elif name == "log_joint_torques":
            nc.log_joint_torques(payload, timestamp=timestamp)
        elif name == "log_joint_target_positions":
            nc.log_joint_target_positions(payload, timestamp=timestamp)
        elif name == "log_parallel_gripper_open_amounts":
            nc.log_parallel_gripper_open_amounts(payload, timestamp=timestamp)
        elif name == "log_parallel_gripper_target_open_amounts":
            nc.log_parallel_gripper_target_open_amounts(payload, timestamp=timestamp)
        elif name == "log_rgb":
            camera_name = next(iter(payload))
            image_array = payload[camera_name]
            nc.log_rgb(camera_name, image_array, timestamp=timestamp)
        else:
            print(f"\n⚠️  Unknown logging function: {name}")
    except Exception as e:
        print(f"\n⚠️  Failed to call {name} to Neuracore. Exception: {e}")
        print("Traceback:")
        traceback.print_exc()


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
            play_audio_feedback("start")  # <-- Trigger distinct START sound
        except Exception as e:
            print(f"✗ Failed to start recording. Exception: {e}")
            print("Traceback:")
            traceback.print_exc()
    else:
        # Stop recording
        try:
            nc.stop_recording()
            print("✓ ⏹️ Recording stopped (Button RJ)")
            play_audio_feedback("stop")  # <-- Trigger distinct STOP sound
        except Exception as e:
            print(f"✗ Failed to stop recording. Exception: {e}")
            print("Traceback:")
            traceback.print_exc()


def _step_wrist_joint(delta_degrees: float) -> None:
    """Apply a relative step to the wrist joint target angle."""
    # Prevent IK teleop loop from overwriting this manual joint adjustment.
    data_manager.set_teleop_state(False, None, None)
    robot_controller.set_control_mode(PiperController.ControlMode.JOINT_SPACE)

    target_joint_angles = robot_controller.get_target_joint_angles()
    current_joint_angles = data_manager.get_current_joint_angles()
    if current_joint_angles is not None:
        base_wrist_joint_angle = float(current_joint_angles[-1])
    else:
        base_wrist_joint_angle = float(target_joint_angles[-1])

    target_joint_angles[-1] = base_wrist_joint_angle + delta_degrees
    robot_controller.set_target_joint_angles(target_joint_angles)
    data_manager.set_target_joint_angles(target_joint_angles)


def on_button_y_pressed() -> None:
    """Handle Button Y press to add +5° on the wrist joint."""
    robot_activity_state = data_manager.get_robot_activity_state()
    if robot_activity_state != RobotActivityState.ENABLED:
        print("⚠️  Button Y pressed but robot is not enabled")
        return
    _step_wrist_joint(WRIST_JOINT_BUTTON_STEP_DEGREES)


def on_button_x_pressed() -> None:
    """Handle Button X press to subtract -5° on the wrist joint."""
    robot_activity_state = data_manager.get_robot_activity_state()
    if robot_activity_state != RobotActivityState.ENABLED:
        print("⚠️  Button X pressed but robot is not enabled")
        return
    _step_wrist_joint(-WRIST_JOINT_BUTTON_STEP_DEGREES)


def on_button_lj_pressed() -> None:
    """Toggle slow translation/rotation scaling mode from config values."""
    slow_scaling_mode_enabled = data_manager.toggle_slow_scaling_mode_enabled()
    if slow_scaling_mode_enabled:
        data_manager.set_teleop_scaling(SLOW_TRANSLATION_SCALE, SLOW_ROTATION_SCALE)
        print(
            "🐢 Button LJ pressed - Slow scaling enabled "
            f"(translation={SLOW_TRANSLATION_SCALE:.3f}, "
            f"rotation={SLOW_ROTATION_SCALE:.3f})"
        )
    else:
        data_manager.set_teleop_scaling(TRANSLATION_SCALE, ROTATION_SCALE)
        print(
            "🐇 Button LJ pressed - Slow scaling disabled "
            f"(translation={TRANSLATION_SCALE:.3f}, "
            f"rotation={ROTATION_SCALE:.3f})"
        )


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
    data_manager.set_teleop_scaling(TRANSLATION_SCALE, ROTATION_SCALE)

    # Initialize robot controller
    print("\n🤖 Initializing Piper robot controller...")
    robot_controller = PiperController(
        can_interface="can0",
        robot_rate=ROBOT_RATE,
        control_mode=PiperController.ControlMode.JOINT_SPACE,
        neutral_joint_angles=NEUTRAL_JOINT_ANGLES,
        neutral_end_effector_pose=NEUTRAL_END_EFFECTOR_POSE,
        enable_joint_angle_limits=False,
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
        axis_mask=META_QUEST_AXIS_MASK,
    )

    # Register button callbacks (after state and robot_controller are initialized)
    quest_reader.on("button_a_pressed", on_button_a_pressed)
    quest_reader.on("button_b_pressed", on_button_b_pressed)
    quest_reader.on("button_rj_pressed", on_button_rj_pressed)
    quest_reader.on("button_y_pressed", on_button_y_pressed)
    quest_reader.on("button_x_pressed", on_button_x_pressed)
    quest_reader.on("button_lj_pressed", on_button_lj_pressed)

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

    # Start cameras threads
    print("\n📷 Starting cameras threads...")
    realsense_camera_thread_obj = threading.Thread(
        target=realsense_camera_thread, args=(data_manager,), daemon=True
    )
    realsense_camera_thread_obj.start()

    # usb_camera_thread_obj = threading.Thread(
    #     target=usb_camera_thread, args=(data_manager,), daemon=True
    # )
    # usb_camera_thread_obj.start()

    print()
    print("🚀 Starting teleoperation with REAL ROBOT CONTROL...")
    print("🎮 CONTROLS:")
    print("   1. Press BUTTON A to enable/disable robot")
    print("   2. Hold RIGHT GRIP to activate teleoperation")
    print("   3. Move controller - robot follows!")
    print("   4. Hold RIGHT TRIGGER to close gripper")
    print("   5. Press BUTTON B to send robot home")
    print("   6. Press RIGHT JOYSTICK to start/stop data recording")
    print("   7. Press BUTTON Y to add +5° on the wrist joint")
    print("   8. Press BUTTON X to subtract 5° on the wrist joint")
    print("   9. Press LEFT JOYSTICK BUTTON to toggle slow scaling mode")
    print("   10. Release grip to stop")
    print("⚠️  Press Ctrl+C to exit")
    print()

    try:
        while not data_manager.is_shutdown_requested():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Interrupt received - shutting down gracefully...")
        data_manager.request_shutdown()
    except Exception as e:
        print(f"\n❌ Demo error. Exception: {e}")
        print("Traceback:")
        traceback.print_exc()
        data_manager.request_shutdown()
    else:
        print("\n⚠️  Worker requested shutdown - cleaning up...")
    # Cleanup
    print("\n🧹 Cleaning up...")

    # Cancel recording if active
    if nc.is_recording():
        try:
            print("⚠️  Cancelling active recording...")
            nc.cancel_recording()
            print("✓ Recording cancelled")
        except Exception as e:
            print(f"⚠️  Error cancelling recording. Exception: {e}")
            print("Traceback:")
            traceback.print_exc()

    # shutdown threads and data producers before logging out
    data_manager.request_shutdown()
    data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
    quest_reader.stop()
    quest_thread.join()
    joint_state_thread_obj.join()
    ik_thread.join()
    realsense_camera_thread_obj.join()
    # usb_camera_thread_obj.join()
    nc.logout()
    robot_controller.cleanup()

    print("\n👋 Demo stopped.")
