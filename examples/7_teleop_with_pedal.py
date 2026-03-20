#!/usr/bin/env python3
"""Piper Robot Teleoperation with Meta Quest and Foot Pedal control.

This script combines Meta Quest controller input for movement with Foot Pedal
control for session management (Activate, Home, Record).
"""

import argparse
import multiprocessing
import sys
import threading
import time
from pathlib import Path
from typing import Any

import neuracore as nc
import numpy as np

# Add parent directory to path to import pink_ik_solver and piper_controller
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import (
    CONTROLLER_BETA,
    CONTROLLER_D_CUTOFF,
    CONTROLLER_MIN_CUTOFF,
    GRIPPER_FRAME_NAME,
    GRIPPER_LOGGING_NAME,
    JOINT_NAMES,
    NEUTRAL_JOINT_ANGLES,
    POSTURE_COST_VECTOR,
    ROBOT_RATE,
    URDF_PATH,
)
from common.data_manager import DataManager, RobotActivityState
from common.foot_pedal import FootPedal
from common.threads.camera import camera_thread
from common.threads.ik_solver import ik_solver_thread
from common.threads.joint_state import joint_state_thread
from common.threads.quest_reader import quest_reader_thread
from meta_quest_teleop.reader import MetaQuestReader

from pink_ik_solver import PinkIKSolver
from piper_controller import PiperController

ENABLE_DISABLE_PEDAL = "a"
HOME_POSITION_PEDAL = "b"
RECORD_TOGGLE_PEDAL = "c"


def log_to_neuracore_on_change_callback(
    name: str, value: Any, timestamp: float
) -> None:
    """Callback triggered on state changes to log data to Neuracore.

    Args:
        name: Name of the data stream.
        value: Data value (float, array, or image).
        timestamp: Time of the change.
    """
    try:
        if name == "log_joint_positions":
            data_dict = {jn: np.radians(a) for jn, a in zip(JOINT_NAMES, value)}
            nc.log_joint_positions(data_dict, timestamp=timestamp)
        elif name == "log_joint_target_positions":
            data_dict = {jn: np.radians(a) for jn, a in zip(JOINT_NAMES, value)}
            nc.log_joint_target_positions(data_dict, timestamp=timestamp)
        elif name == "log_parallel_gripper_open_amounts":
            nc.log_parallel_gripper_open_amounts(
                {GRIPPER_LOGGING_NAME: value}, timestamp=timestamp
            )
        elif name == "log_parallel_gripper_target_open_amounts":
            nc.log_parallel_gripper_target_open_amounts(
                {GRIPPER_LOGGING_NAME: value}, timestamp=timestamp
            )
        elif name == "log_rgb":
            nc.log_rgb("rgb", value, timestamp=timestamp)
    except Exception as e:
        print(f"⚠️  Logging failed: {e}")


def toggle_robot_state() -> None:
    """Toggle the robot's activity state between ENABLED and DISABLED."""
    print("🔘 Pedal toggled - Robot State")
    state = data_manager.get_robot_activity_state()

    if state == RobotActivityState.ENABLED:
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        if robot_controller:
            robot_controller.graceful_stop()
        data_manager.set_teleop_state(False, None, None)
        print("✓ 🔴 Robot disabled (Pedal)")
    elif state == RobotActivityState.DISABLED or state == RobotActivityState.HOMING:
        # If no robot, just toggle the state for dashboard visibility
        if not robot_controller:
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
            print("✓ 🟢 Robot enabled (Pedal - Headless)")
            return

        if robot_controller.resume_robot():
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
            print("✓ 🟢 Robot enabled (Pedal)")
        else:
            print("✗ [ACTION] Failed to enable robot")


def move_robot_home() -> None:
    """Command the robot to move to its neutral/home position."""
    print("🏠 Pedal toggled - Move Home")
    state = data_manager.get_robot_activity_state()

    if state == RobotActivityState.ENABLED:
        print("🏠 Pedal pressed - Moving to home position...")
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
        data_manager.set_teleop_state(False, None, None)

        if robot_controller:
            if not robot_controller.move_to_home():
                print("✗ Failed to initiate home move")
                data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
        else:
            # Headless simulation of homing
            time.sleep(1)
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
            print("✓ 🏠 Robot homed (Headless)")
    else:
        print("⚠️  Pedal pressed but robot is not enabled")


def toggle_recording() -> None:
    """Start or stop a data recording session in Neuracore."""
    print("⏺️ Pedal toggled - Recording")
    if not nc.is_recording():
        try:
            nc.start_recording()
            print("✓ 🔴 Recording started (Pedal)")
        except Exception as e:
            print(f"✗ Failed to start recording: {e}")
    else:
        try:
            nc.stop_recording()
            print("✓ ⏹️ Recording stopped (Pedal)")
        except Exception as e:
            print(f"✗ Failed to stop recording: {e}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Combined Quest + Pedal Teleoperation")
    parser.add_argument("--ip-address", type=str, help="Meta Quest IP")
    parser.add_argument("--dataset-name", type=str, help="Neuracore dataset name")
    args = parser.parse_args()

    print("=" * 60)
    print("PIPER TELEOP: META QUEST + FOOT PEDALS")
    print("=" * 60)

    # Neuracore Init
    print("\n🔧 Initializing Neuracore...")
    try:
        nc.login()
        nc.connect_robot(
            robot_name="AgileX PiPER", urdf_path=str(URDF_PATH), overwrite=False
        )
        ds_name = args.dataset_name or f"pedal-teleop-{time.strftime('%H-%M-%S')}"
        nc.create_dataset(name=ds_name, description="Quest + Pedal unified collection")
    except Exception as e:
        print(f"⚠️  Neuracore initialization skipped/failed: {e}")

    # Shared State
    data_manager = DataManager()
    data_manager.set_on_change_callback(log_to_neuracore_on_change_callback)
    data_manager.set_controller_filter_params(
        CONTROLLER_MIN_CUTOFF, CONTROLLER_BETA, CONTROLLER_D_CUTOFF
    )

    # Robot Initialization
    print("\n🤖 Initializing Piper...")
    robot_controller = None
    try:
        robot_controller = PiperController(can_interface="can0", robot_rate=ROBOT_RATE)
        robot_controller.start_control_loop()
    except Exception as e:
        print(f"⚠️  Robot controller initialization skipped/failed: {e}")

    # Threads
    print("\n📊 Starting Threads (JointState, QuestReader, IKSolver, Camera)...")
    pedal_thread = None
    if robot_controller:
        threading.Thread(
            target=joint_state_thread,
            args=(data_manager, robot_controller),
            daemon=True,
        ).start()

    quest_reader = None
    try:
        print("🔍 Searching for Meta Quest...")
        # Adb initialization in the reader might call sys.exit(1), so we catch BaseException
        quest_reader = MetaQuestReader(ip_address=args.ip_address, port=5555, run=True)
        threading.Thread(
            target=quest_reader_thread, args=(data_manager, quest_reader), daemon=True
        ).start()
    except (Exception, BaseException) as e:
        print(f"⚠️  Quest reader initialization skipped/failed: {e}")

    # Sync IK solver to current position
    try:
        initial_joints = np.radians(
            data_manager.get_current_joint_angles() or NEUTRAL_JOINT_ANGLES
        )
        ik_solver = PinkIKSolver(
            urdf_path=URDF_PATH,
            end_effector_frame=GRIPPER_FRAME_NAME,
            initial_configuration=initial_joints,
            posture_cost_vector=np.array(POSTURE_COST_VECTOR),
        )
        threading.Thread(
            target=ik_solver_thread, args=(data_manager, ik_solver), daemon=True
        ).start()
    except Exception as e:
        print(f"⚠️  IK Solver initialization skipped/failed: {e}")

    try:
        threading.Thread(
            target=camera_thread, args=(data_manager,), daemon=True
        ).start()
    except Exception as e:
        print(f"⚠️  Camera thread failed to start: {e}")

    # Foot Pedal – started as a daemon thread, callbacks wired inline
    print("\n⌨️  Initializing Foot Pedals...")
    pedal = FootPedal(
        key_map={
            "button_a": ENABLE_DISABLE_PEDAL,
            "button_b": HOME_POSITION_PEDAL,
            "button_c": RECORD_TOGGLE_PEDAL,
        },
    )
    pedal.bind("button_a", toggle_robot_state)
    pedal.bind("button_b", move_robot_home)
    pedal.bind("button_c", toggle_recording)

    pedal_thread = threading.Thread(
        target=pedal.run_loop, args=(data_manager,), daemon=True
    )
    pedal_thread.start()

    print("\n✅ SYSTEM ONLINE")
    print("------------------------------------------------------------")
    print("🎮 QUEST CONTROLS:  Hold GRIP to move, TRIGGER for gripper")
    print("⌨️  PEDAL CONTROLS:  ENABLE/DISABLE (A), HOME (B), RECORD (C)")
    print("------------------------------------------------------------")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        if pedal_thread:
            pedal_thread.join(timeout=1.0)
        try:
            if nc.is_recording():
                nc.cancel_recording()
            nc.logout()
        except Exception:
            pass
        data_manager.request_shutdown()
        if quest_reader:
            try:
                quest_reader.stop()
            except Exception:
                pass
        if robot_controller:
            try:
                robot_controller.cleanup()
            except Exception:
                pass
