#!/usr/bin/env python3
"""Piper Robot Teleoperation and Neuracore Data Collection.

This script facilitates real-time teleoperation of the AgileX Piper robotic arm
using Meta Quest controllers, while simultaneously streaming perfectly synchronized
telemetry data (joint states, RGB camera frames, gripper states) to the Neuracore cloud.

Key Features:
    - Connects to Neuracore and initializes a new dataset session.
    - Bootstraps background threads for hardware control (CAN bus), IK solving, and camera streaming.
    - Initializes the Meta Quest ADB bridge for VR controller tracking.
    - Implements robust error handling for common ADB/USB debugging permission failures.
    - Maps physical Quest controller buttons to robot state actions (Home, Enable, Record).
    - Dynamically loads scaling and IK parameters from a YAML config file.

Hardware Requirements:
    - AgileX Piper robot arm connected via CAN interface ('can0').
    - RealSense camera or USB WebCam configured in `common.configs`.
    - Meta Quest headset connected via USB with 'USB Debugging' explicitly allowed.

Usage:
    python 2_collect_teleop_data_with_neuracore.py --ip-address <QUEST_IP> --dataset-name <OPTIONAL_NAME> --ik-config <PATH_TO_YAML>
"""

import argparse
import multiprocessing
import sys
import threading
import time
from pathlib import Path

import neuracore as nc

# ---------------------------------------------------------------------------
# Path Configuration & Local Imports
# ---------------------------------------------------------------------------
# Append parent directory to sys.path to resolve local 'common' modules.
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config_parser import load_ik_config
from common.configs import META_QUEST_AXIS_MASK, URDF_PATH
from common.data_manager import DataManager, RobotActivityState
from common.shared_actions import (
    move_robot_home,
    neuracore_logging_callback,
    toggle_recording,
    toggle_robot_enabled,
)
from common.system_bootstrap import bootstrap_robot_system
from common.threads.quest_reader import quest_reader_thread
from meta_quest_teleop.reader import MetaQuestReader

from piper_controller import PiperController

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _step_wrist_joint(
    data_manager: DataManager, robot_controller: PiperController, delta_degrees: float
) -> None:
    """Apply a relative step adjustment to the robot's wrist joint target angle."""
    data_manager.set_teleop_state(False, None, None)
    robot_controller.set_control_mode(PiperController.ControlMode.JOINT_SPACE)

    target_joint_angles = robot_controller.get_target_joint_angles()
    current_joint_angles = data_manager.get_current_joint_angles()

    # Fallback to target angles if hardware telemetry is temporarily unavailable
    base_wrist = (
        float(current_joint_angles[-1])
        if current_joint_angles is not None
        else float(target_joint_angles[-1])
    )

    target_joint_angles[-1] = base_wrist + delta_degrees
    robot_controller.set_target_joint_angles(target_joint_angles)
    data_manager.set_target_joint_angles(target_joint_angles)


def toggle_slow_scaling(data_manager: DataManager, tele_p: dict) -> None:
    """Toggle the teleoperation sensitivity between standard and slow (precision) scaling."""
    enabled = data_manager.toggle_slow_scaling_mode_enabled()
    if enabled:
        data_manager.set_teleop_scaling(
            tele_p.get("slow_translation_scale", 0.6),
            tele_p.get("slow_rotation_scale", 0.6),
        )
        print("🐢 Slow scaling (precision mode) enabled")
    else:
        data_manager.set_teleop_scaling(
            tele_p.get("translation_scale", 1.5), tele_p.get("rotation_scale", 1.2)
        )
        print("🐇 Slow scaling disabled (standard mode)")


if __name__ == "__main__":
    # Ensure safe multiprocess spawning for UI/Background threads across different OS environments
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        description="Teleop with Neuracore Data Collection"
    )
    parser.add_argument(
        "--ip-address",
        type=str,
        default=None,
        help="IP address of the Meta Quest headset.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Override the auto-generated dataset name.",
    )
    parser.add_argument(
        "--ik-config",
        type=str,
        default="ik_conf/default.yaml",
        help="Path to IK/teleop YAML config.",
    )
    args = parser.parse_args()

    print("=" * 60 + "\nPIPER TELEOP DATA COLLECTION\n" + "=" * 60)

    # Load the YAML configuration dictionary
    config = load_ik_config(args.ik_config)
    tele_p = config.get("teleop_parameters", {})
    wrist_step = tele_p.get("wrist_joint_button_step_degrees", 5.0)

    # ---------------------------------------------------------
    # 1. Neuracore Initialization & Dataset Creation
    # ---------------------------------------------------------
    print("\n🔧 Initializing Neuracore...")
    nc.login()
    nc.connect_robot(
        robot_name="AgileX PiPER", urdf_path=str(URDF_PATH), overwrite=False
    )

    # Auto-generate a timestamped dataset name if one was not provided via CLI
    dataset_name = (
        args.dataset_name or f"piper-teleop-data-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    nc.create_dataset(
        name=dataset_name, description="Teleop data collection for Piper robot"
    )

    # ---------------------------------------------------------
    # 2. Hardware & Subsystem Bootstrapping
    # ---------------------------------------------------------
    data_manager, robot_controller, ik_solver, active_threads = bootstrap_robot_system(
        config, start_ik=True, start_camera=True
    )

    # Bind the unified Neuracore logger so state changes automatically push to the cloud
    data_manager.set_on_change_callback(neuracore_logging_callback)

    # ---------------------------------------------------------
    # 3. Meta Quest Initialization & Error Handling
    # ---------------------------------------------------------
    print("\n🎮 Initializing Meta Quest reader...")
    try:
        quest_reader = MetaQuestReader(
            ip_address=args.ip_address,
            port=5555,
            run=True,
            axis_mask=META_QUEST_AXIS_MASK,
        )
    except (Exception, SystemExit):
        print("\n" + "!" * 60)
        print("❌ FAILED TO ACCESS META QUEST")
        print("!" * 60)
        print("The headset is plugged in, but ADB debugging permissions are missing.")
        print("\nPLEASE FOLLOW THESE STEPS:")
        print("  1. Put the Meta Quest headset on your head.")
        print("  2. Look for a notification in your menu that says 'USB Detected'.")
        print(
            "  3. Click on that notification and select 'Allow' to grant data access."
        )
        print("  4. Rerun this script.")
        print("!" * 60 + "\n")

        data_manager.request_shutdown()
        for t in active_threads:
            t.join(timeout=1.0)
        robot_controller.cleanup()
        nc.logout()
        sys.exit(1)

    # ---------------------------------------------------------
    # 4. Input Binding & Background Threading
    # ---------------------------------------------------------
    quest_reader.on(
        "button_a_pressed", lambda: toggle_robot_enabled(data_manager, robot_controller)
    )
    quest_reader.on(
        "button_b_pressed", lambda: move_robot_home(data_manager, robot_controller)
    )
    quest_reader.on("button_rj_pressed", lambda: toggle_recording(play_audio=True))
    quest_reader.on(
        "button_y_pressed",
        lambda: _step_wrist_joint(data_manager, robot_controller, wrist_step),
    )
    quest_reader.on(
        "button_x_pressed",
        lambda: _step_wrist_joint(data_manager, robot_controller, -wrist_step),
    )
    quest_reader.on(
        "button_lj_pressed", lambda: toggle_slow_scaling(data_manager, tele_p)
    )

    quest_thread = threading.Thread(
        target=quest_reader_thread, args=(data_manager, quest_reader), daemon=True
    )
    quest_thread.start()
    active_threads.append(quest_thread)

    print(
        "\n🚀 Ready! Use Meta Quest controllers. Press Right Joystick to Record. Press Ctrl+C to exit.\n"
    )

    # ---------------------------------------------------------
    # 5. Main Daemon Loop
    # ---------------------------------------------------------
    try:
        while not data_manager.is_shutdown_requested():
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n👋 Interrupt received - shutting down gracefully...")
        data_manager.request_shutdown()
    except Exception as e:
        print(f"\n❌ Unhandled Demo error: {e}")
        data_manager.request_shutdown()

    # ---------------------------------------------------------
    # 6. Graceful Teardown & Cleanup
    # ---------------------------------------------------------
    print("\n🧹 Cleaning up subsystems...")

    if nc.is_recording():
        nc.cancel_recording()

    data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
    quest_reader.stop()

    for t in active_threads:
        t.join(timeout=2.0)

    nc.logout()
    robot_controller.cleanup()
    print("👋 Demo stopped.")
