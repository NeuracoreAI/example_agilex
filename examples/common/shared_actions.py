import subprocess
from typing import Any

import neuracore as nc
from common.data_manager import DataManager, RobotActivityState

from piper_controller import PiperController


def toggle_robot_enabled(
    data_manager: DataManager, robot_controller: PiperController, visualizer=None
):
    """Safely toggles the robot between ENABLED and DISABLED states."""
    state = data_manager.get_robot_activity_state()

    if state == RobotActivityState.ENABLED:
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        if robot_controller:
            robot_controller.graceful_stop()
        data_manager.set_teleop_state(False, None, None)
        if visualizer:
            visualizer.update_toggle_robot_enabled_status(False)
        print("✓ 🔴 Robot disabled")

    elif state in (RobotActivityState.DISABLED, RobotActivityState.HOMING):
        if not robot_controller:
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
            print("✓ 🟢 Robot enabled (Headless)")
            return

        if robot_controller.resume_robot():
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
            if visualizer:
                visualizer.update_toggle_robot_enabled_status(True)
            print("✓ 🟢 Robot enabled")
        else:
            print("✗ Failed to enable robot")


def move_robot_home(data_manager: DataManager, robot_controller: PiperController):
    """Safely commands the robot to return to its home position."""
    state = data_manager.get_robot_activity_state()

    if state == RobotActivityState.ENABLED:
        print("🏠 Moving to home position...")
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
        data_manager.set_teleop_state(False, None, None)

        if robot_controller:
            if not robot_controller.move_to_home():
                print("✗ Failed to initiate home move")
                data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
        else:
            print("✓ 🏠 Robot homed (Headless)")
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
    else:
        print("⚠️ Robot is not enabled, cannot go home")


def toggle_recording(play_audio: bool = False) -> None:
    """Safely start or stop a Neuracore data recording session."""
    if not nc.is_recording():
        try:
            nc.start_recording()
            print("✓ 🔴 Recording started")
            if play_audio:
                subprocess.Popen(
                    ["play", "-q", "-n", "synth", "0.3", "sine", "880"],
                    stderr=subprocess.DEVNULL,
                )
        except Exception as e:
            print(f"✗ Failed to start recording: {e}")
    else:
        try:
            nc.stop_recording()
            print("✓ ⏹️ Recording stopped")
            if play_audio:
                subprocess.Popen(
                    ["play", "-q", "-n", "synth", "0.3", "sine", "440"],
                    stderr=subprocess.DEVNULL,
                )
        except Exception as e:
            print(f"✗ Failed to stop recording: {e}")


def neuracore_logging_callback(
    name: str, payload: dict[str, Any], timestamp: float
) -> None:
    """Unified callback to map DataManager state changes to Neuracore log streams."""
    try:
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
            nc.log_rgb(camera_name, payload[camera_name], timestamp=timestamp)
    except Exception as e:
        print(f"⚠️ Logging failed for {name}: {e}")
