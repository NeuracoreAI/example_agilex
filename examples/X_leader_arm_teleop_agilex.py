#!/usr/bin/env python3
"""Piper (Agilex) teleop with SO100 leader arm - direct joint mapping.

Leader arm drives the Piper URDF in the GUI; optionally drive the real robot
with --real-robot. The leader arm class is configured with Piper follower
limits, offsets, and directions and outputs mapped joint angles + gripper.

Requirements:
- Leader arm: calibrated S0100 (use same --leader-id as when calibrating).
- Conda: piper-teleop.

Usage:
  cd example_agilex/examples
  python 0_leader_arm_teleop_agilex.py --leader-port /dev/ttyACM0 --leader-id my_awesome_leader_arm
  python 0_leader_arm_teleop_agilex.py --real-robot --leader-port /dev/ttyACM0 --leader-id my_awesome_leader_arm

  Move the leader arm to drive the URDF (and robot if --real-robot). Ctrl+C to exit.
"""

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import NEUTRAL_JOINT_ANGLES, ROBOT_RATE, VISUALIZATION_RATE
from common.data_manager import DataManager, RobotActivityState
from common.leader_arm import SO100LeaderArm
from common.robot_visualizer import RobotVisualizer
from common.threads.leader_reader import leader_reader_thread

# NOTE:Piper follower config for the leader arm (limits, offsets, directions, mapping).
# Leader 5 DOF + gripper -> Piper 6 DOF + gripper; Piper joint 4 is fixed at 0.

PIPER_JOINT_LIMITS_DEG = np.array(
    [
        (-150.0, 150.0),
        (0.0, 180.0),
        (-170.0, 0.0),
        (-100.0, 100.0),
        (-70.0, 70.0),
        (-120.0, 120.0),
    ],
    dtype=np.float64,
)

# NOTE: Measure leader zero vs Piper zero per joint (degrees). joint4 unused.
PIPER_OFFSETS_DEG = np.array(
    [0.0, 90.0, -90.0, 0.0, 0.0, -45.0],
    dtype=np.float64,
)

PIPER_DIRECTIONS = np.array(
    [-1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
    dtype=np.float64,
)

# Leader joint index -> Piper joint index. Piper joint 4 (index 3) has no leader.
LEADER_TO_PIPER_JOINT = {0: 0, 1: 1, 2: 2, 3: 4, 4: 5}
PIPER_FIXED_JOINTS = {3: 0.0}


def main() -> None:
    """Run Piper teleop with SO100 leader arm (URDF and optionally real robot)."""
    from common.configs import URDF_PATH

    parser = argparse.ArgumentParser(
        description="Piper URDF teleop with SO100 leader arm."
    )
    parser.add_argument("--leader-port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--leader-id", type=str, default="my_awesome_leader_arm")
    parser.add_argument("--leader-rate", type=float, default=50.0)
    parser.add_argument(
        "--real-robot",
        action="store_true",
        help="Use the real Piper robot (default: URDF only)",
    )
    parser.add_argument("--can-interface", type=str, default="can0")
    args = parser.parse_args()

    use_real_robot = args.real_robot
    print("=" * 60)
    print(
        "PIPER TELEOP WITH SO100 LEADER ARM"
        + (" - REAL ROBOT" if use_real_robot else " - URDF only")
    )
    print("=" * 60)

    leader = SO100LeaderArm(port=args.leader_port, calibration_id=args.leader_id)
    leader.configure_follower(
        follower_limits_deg=PIPER_JOINT_LIMITS_DEG,
        follower_offsets_deg=PIPER_OFFSETS_DEG,
        follower_directions=PIPER_DIRECTIONS,
        leader_to_follower_joint=LEADER_TO_PIPER_JOINT,
        fixed_joints=PIPER_FIXED_JOINTS,
    )
    print("\n🦾 Connecting to S0100 leader arm...")
    try:
        leader.connect(calibrate=False)
    except Exception as e:
        print(f"Failed to connect to leader: {e}")
        if "no calibration registered" in str(e):
            print(
                "Create a calibration JSON for this leader id under "
                "~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/"
            )
        sys.exit(1)
    print("✓ Leader arm connected")

    data_manager = DataManager()

    robot_controller = None
    joint_state_thread_obj = None
    if use_real_robot:
        from common.threads.joint_state import joint_state_thread

        from piper_controller import PiperController

        print("\n🤖 Initializing Piper robot controller...")
        robot_controller = PiperController(
            can_interface=args.can_interface,
            robot_rate=ROBOT_RATE,
            control_mode=PiperController.ControlMode.JOINT_SPACE,
            neutral_joint_angles=NEUTRAL_JOINT_ANGLES,
            debug_mode=False,
        )
        robot_controller.start_control_loop()
        print("📊 Starting joint state thread...")
        joint_state_thread_obj = threading.Thread(
            target=joint_state_thread,
            args=(data_manager, robot_controller),
            daemon=True,
        )
        joint_state_thread_obj.start()

    leader_thread = threading.Thread(
        target=leader_reader_thread,
        args=(data_manager, leader, args.leader_rate),
        daemon=True,
    )
    leader_thread.start()

    visualizer = RobotVisualizer(urdf_path=URDF_PATH)
    visualizer.add_basic_controls()
    visualizer.add_teleop_controls()
    visualizer.add_gripper_status_controls()
    if use_real_robot:
        visualizer.add_homing_controls()
        visualizer.add_toggle_robot_enabled_status_button()

        def toggle_robot_enabled() -> None:
            assert robot_controller is not None  # Only registered when use_real_robot
            state = data_manager.get_robot_activity_state()
            if state == RobotActivityState.ENABLED:
                data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
                robot_controller.graceful_stop()
                data_manager.set_teleop_state(False, None, None)
                visualizer.update_toggle_robot_enabled_status(False)
                print("✓ 🔴 Robot disabled")
            elif state == RobotActivityState.DISABLED:
                if robot_controller.resume_robot():
                    data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
                    visualizer.update_toggle_robot_enabled_status(True)
                    print("✓ 🟢 Robot enabled")
                else:
                    print("✗ Failed to enable robot")

        def on_go_home() -> None:
            assert robot_controller is not None  # Only registered when use_real_robot
            state = data_manager.get_robot_activity_state()
            if state == RobotActivityState.ENABLED:
                data_manager.set_robot_activity_state(RobotActivityState.HOMING)
                data_manager.set_teleop_state(False, None, None)
                ok = robot_controller.move_to_home()
                if not ok:
                    data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
            else:
                print("⚠️ Cannot home: robot not enabled")

        visualizer.set_toggle_robot_enabled_status_callback(toggle_robot_enabled)
        visualizer.set_go_home_callback(on_go_home)

    print()
    if use_real_robot:
        print(
            "🚀 Leader arm driving REAL Piper robot. Enable robot in GUI, then move leader. Ctrl+C to exit."
        )
    else:
        print("🚀 Leader arm driving Piper URDF. Move the leader arm. Ctrl+C to exit.")
    print()

    dt_viz = 1.0 / VISUALIZATION_RATE
    try:
        while True:
            t0 = time.time()
            mapped_angles, mapped_gripper = data_manager.get_leader_mapped_state()
            if mapped_angles is not None and mapped_gripper is not None:
                data_manager.set_target_joint_angles(mapped_angles)
                data_manager.set_controller_data(None, 1.0, 1.0 - mapped_gripper)
                data_manager.set_teleop_state(True, None, None)
                if not use_real_robot:
                    data_manager.set_current_joint_angles(mapped_angles)

            current_joint_angles = data_manager.get_current_joint_angles()
            _, grip_value, trigger_value = data_manager.get_controller_data()
            target_joint_angles = data_manager.get_target_joint_angles()

            visualizer.set_grip_value(grip_value)
            visualizer.set_trigger_value(trigger_value)
            visualizer.update_teleop_status(data_manager.get_teleop_active())

            if use_real_robot:
                robot_activity_state = data_manager.get_robot_activity_state()
                if robot_activity_state == RobotActivityState.ENABLED:
                    visualizer.update_robot_status("Robot Status: Enabled")
                elif robot_activity_state == RobotActivityState.HOMING:
                    visualizer.update_robot_status("Robot Status: Homing")
                else:
                    visualizer.update_robot_status("Robot Status: Disabled")
                if (
                    target_joint_angles is not None
                    and robot_activity_state == RobotActivityState.ENABLED
                ):
                    visualizer.update_ghost_robot_visibility(True)
                    visualizer.update_ghost_robot_pose(np.radians(target_joint_angles))
                else:
                    visualizer.update_ghost_robot_visibility(False)
                visualizer.update_gripper_status(
                    trigger_value,
                    robot_enabled=(robot_activity_state == RobotActivityState.ENABLED),
                )
            else:
                visualizer.update_robot_status("URDF only – leader arm driving")
                visualizer.update_ghost_robot_visibility(False)
                visualizer.update_gripper_status(trigger_value, robot_enabled=True)

            if current_joint_angles is not None:
                current_rad = np.radians(current_joint_angles)
                visualizer.update_robot_pose(current_rad)
                visualizer.update_joint_angles_display(current_rad)

            elapsed = time.time() - t0
            if dt_viz - elapsed > 0:
                time.sleep(dt_viz - elapsed)

    except KeyboardInterrupt:
        print("\n\n👋 Interrupt – shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()

    print("\n🧹 Cleaning up...")
    data_manager.request_shutdown()
    data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
    leader_thread.join(timeout=2.0)
    if joint_state_thread_obj is not None:
        joint_state_thread_obj.join(timeout=2.0)
    if robot_controller is not None:
        robot_controller.cleanup()
    leader.disconnect()
    visualizer.stop()
    print("👋 Done.")


if __name__ == "__main__":
    main()
