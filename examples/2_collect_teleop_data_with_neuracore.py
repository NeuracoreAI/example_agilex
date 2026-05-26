#!/usr/bin/env python3
"""Piper Robot Teleoperation with Meta Quest Controller and Neuracore data collection."""

import argparse
import multiprocessing
import sys
import threading
import time
import traceback
from pathlib import Path
import numpy as np
import neuracore as nc

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import (
    META_QUEST_AXIS_MASK, URDF_PATH, WRIST_JOINT_BUTTON_STEP_DEGREES,
    SLOW_TRANSLATION_SCALE, SLOW_ROTATION_SCALE, TRANSLATION_SCALE, ROTATION_SCALE
)
from common.data_manager import RobotActivityState
from common.system_bootstrap import bootstrap_robot_system
from common.shared_actions import (
    toggle_robot_enabled, move_robot_home, 
    toggle_recording, neuracore_logging_callback
)
from common.threads.quest_reader import quest_reader_thread
from meta_quest_teleop.reader import MetaQuestReader
from piper_controller import PiperController


def _step_wrist_joint(data_manager, robot_controller, delta_degrees: float) -> None:
    """Apply a relative step to the wrist joint target angle."""
    data_manager.set_teleop_state(False, None, None)
    robot_controller.set_control_mode(PiperController.ControlMode.JOINT_SPACE)

    target_joint_angles = robot_controller.get_target_joint_angles()
    current_joint_angles = data_manager.get_current_joint_angles()
    base_wrist = float(current_joint_angles[-1]) if current_joint_angles is not None else float(target_joint_angles[-1])

    target_joint_angles[-1] = base_wrist + delta_degrees
    robot_controller.set_target_joint_angles(target_joint_angles)
    data_manager.set_target_joint_angles(target_joint_angles)


def toggle_slow_scaling(data_manager):
    enabled = data_manager.toggle_slow_scaling_mode_enabled()
    if enabled:
        data_manager.set_teleop_scaling(SLOW_TRANSLATION_SCALE, SLOW_ROTATION_SCALE)
        print("🐢 Slow scaling enabled")
    else:
        data_manager.set_teleop_scaling(TRANSLATION_SCALE, ROTATION_SCALE)
        print("🐇 Slow scaling disabled")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(description="Teleop with Neuracore Data Collection")
    parser.add_argument("--ip-address", type=str, default=None, help="Meta Quest IP")
    parser.add_argument("--dataset-name", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60 + "\nPIPER TELEOP DATA COLLECTION\n" + "=" * 60)

    # 1. Connect to Neuracore & Create Dataset
    print("\n🔧 Initializing Neuracore...")
    nc.login()
    nc.connect_robot(robot_name="AgileX PiPER", urdf_path=str(URDF_PATH), overwrite=False)
    
    dataset_name = args.dataset_name or f"piper-teleop-data-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    nc.create_dataset(name=dataset_name, description="Teleop data collection for Piper robot")

    # 2. Bootstrap Core System
    data_manager, robot_controller, ik_solver, active_threads = bootstrap_robot_system(
        start_ik=True, start_camera=True
    )
    
    # Wire Neuracore logging to DataManager
    data_manager.set_on_change_callback(neuracore_logging_callback)

    # 3. Initialize Quest Reader & Bind Controls
    print("\n🎮 Initializing Meta Quest reader...")
    quest_reader = MetaQuestReader(
        ip_address=args.ip_address, port=5555, run=True, axis_mask=META_QUEST_AXIS_MASK
    )

    quest_reader.on("button_a_pressed", lambda: toggle_robot_enabled(data_manager, robot_controller))
    quest_reader.on("button_b_pressed", lambda: move_robot_home(data_manager, robot_controller))
    quest_reader.on("button_rj_pressed", lambda: toggle_recording(play_audio=True))
    quest_reader.on("button_y_pressed", lambda: _step_wrist_joint(data_manager, robot_controller, WRIST_JOINT_BUTTON_STEP_DEGREES))
    quest_reader.on("button_x_pressed", lambda: _step_wrist_joint(data_manager, robot_controller, -WRIST_JOINT_BUTTON_STEP_DEGREES))
    quest_reader.on("button_lj_pressed", lambda: toggle_slow_scaling(data_manager))

    # 4. Start Quest Thread
    quest_thread = threading.Thread(target=quest_reader_thread, args=(data_manager, quest_reader), daemon=True)
    quest_thread.start()
    active_threads.append(quest_thread)

    print("\n🚀 Ready! Use Meta Quest controllers. Press Right Joystick to Record. Press Ctrl+C to exit.\n")

    # 5. Wait Loop (Work happens in background threads)
    try:
        while not data_manager.is_shutdown_requested():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Interrupt received - shutting down gracefully...")
        data_manager.request_shutdown()
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        data_manager.request_shutdown()

    # 6. Cleanup
    print("\n🧹 Cleaning up...")
    if nc.is_recording():
        nc.cancel_recording()
    
    data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
    quest_reader.stop()
    for t in active_threads: 
        t.join()
    nc.logout()
    robot_controller.cleanup()
    print("👋 Demo stopped.")