#!/usr/bin/env python3
"""Visualization of Pink IK solutions with Meta Quest controller input.

This demo visualizes Pink IK solution for a Piper robot on Viser
using Meta Quest controller as an input.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import pink_ik_solver
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add meta_quest_teleop to path
sys.path.insert(0, str(Path(__file__).parent.parent / "meta_quest_teleop"))

import time

import numpy as np
from common.configs import (
    CONTROLLER_BETA,
    CONTROLLER_D_CUTOFF,
    CONTROLLER_MIN_CUTOFF,
    DAMPING_COST,
    FRAME_TASK_GAIN,
    GRIP_THRESHOLD,
    GRIPPER_FRAME_NAME,
    IK_SOLVER_RATE,
    LM_DAMPING,
    NEUTRAL_JOINT_ANGLES,
    ORIENTATION_COST,
    POSITION_COST,
    POSTURE_COST_VECTOR,
    SOLVER_DAMPING_VALUE,
    SOLVER_NAME,
    URDF_PATH,
)
from common.one_euro_filter import OneEuroFilterTransform
from common.robot_visualizer import RobotVisualizer

from meta_quest_teleop.reader import MetaQuestReader
from pink_ik_solver import PinkIKSolver


def main() -> None:
    """Main function for Piper teleoperation visualization demo."""
    parser = argparse.ArgumentParser(
        description="Piper Robot Teleoperation Visualization - NO REAL ROBOT CONTROL"
    )
    parser.add_argument(
        "--ip-address",
        type=str,
        default=None,
        help="IP address of Meta Quest device (optional, defaults to None for auto-discovery)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PIPER ROBOT TELEOPERATION VISUALIZATION (NO REAL ROBOT)")
    print("=" * 60)

    # Initialize Meta Quest reader
    print("\n🎮 Initializing Meta Quest reader...")
    quest_reader = MetaQuestReader(ip_address=args.ip_address, port=5555, run=True)

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
        initial_configuration=np.array(NEUTRAL_JOINT_ANGLES),
        posture_cost_vector=np.array(POSTURE_COST_VECTOR),
    )

    # Set up visualizer
    print("\n🖥️  Starting visualization...")
    visualizer = RobotVisualizer(urdf_path=URDF_PATH)
    visualizer.add_basic_controls()
    visualizer.add_teleop_controls()
    visualizer.add_homing_controls()
    visualizer.add_controller_filter_controls(
        initial_min_cutoff=CONTROLLER_MIN_CUTOFF,
        initial_beta=CONTROLLER_BETA,
        initial_d_cutoff=CONTROLLER_D_CUTOFF,
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
    visualizer.add_grip_threshold_control(initial_threshold=GRIP_THRESHOLD)
    visualizer.add_controller_visualization()
    visualizer.add_target_frame_visualization()

    # Register Button B callback for home position (visualization only)
    def on_button_b_pressed() -> None:
        """Handle Button B press to move robot to home position (visualization)."""
        print("🏠 Button B pressed - Moving to home position (visualization)...")
        visualizer.update_homing_status("Homing: In progress (Button B)")
        # Set IK solver to home configuration
        try:
            ik_solver.set_configuration(np.array(NEUTRAL_JOINT_ANGLES))
            # Update target transform to match home position
            current_end_effector_pose = ik_solver.get_current_end_effector_pose()
            position = current_end_effector_pose[:3, 3]
            orientation = current_end_effector_pose[:3, :3]
            home_target_transform = np.eye(4)
            home_target_transform[:3, :3] = orientation
            home_target_transform[:3, 3] = position
            visualizer.update_homing_status("Homing: Complete (Button B)")
            print("✓ IK solver moved to home position")
        except Exception as e:
            print(f"✗ Failed to move to home: {e}")
            visualizer.update_homing_status("Homing: Failed")

    quest_reader.on("button_b_pressed", on_button_b_pressed)

    print()
    print("🚀 Starting teleoperation visualization...")
    print("📊 VISUALIZATION ONLY - NO REAL ROBOT CONTROL")
    print("🎮 CONTROLS:")
    print("   - Hold RIGHT GRIP to activate teleoperation")
    print("   - Press BUTTON B to move to home position")
    print("   - Release grip to stop")
    print("⚠️  Press Ctrl+C to exit")

    # Teleoperation state tracking
    teleop_active = False
    controller_initial_transform = None
    robot_initial_transform = None
    prev_grip_active = False

    # 1€ Filter for controller smoothing
    controller_filter: OneEuroFilterTransform | None = None

    try:
        while True:
            # Update Meta Quest data
            quest_reader.update()

            # Update Pink parameters
            pink_params = visualizer.get_pink_parameters()
            ik_solver.update_task_parameters(**pink_params)

            # Get controller data
            grip_value = quest_reader.get_grip_value("right")
            visualizer.set_grip_value(grip_value)

            controller_transform_raw = quest_reader.get_hand_controller_transform_ros(
                hand="right"
            )

            # Apply 1€ Filter smoothing to controller transform
            if controller_transform_raw is not None:
                current_time = time.time()

                # Get filter parameters from GUI
                min_cutoff, beta, d_cutoff = visualizer.get_controller_filter_params()

                # Initialize filter if needed
                if controller_filter is None:
                    controller_filter = OneEuroFilterTransform(
                        current_time,
                        controller_transform_raw,
                        min_cutoff,
                        beta,
                        d_cutoff,
                    )
                    controller_transform = controller_transform_raw.copy()
                else:
                    # Update filter parameters if they changed
                    controller_filter.update_params(min_cutoff, beta, d_cutoff)
                    # Apply filter
                    controller_transform = controller_filter(
                        current_time, controller_transform_raw
                    )
            else:
                controller_transform = None
                controller_filter = None

            # Update controller visualization
            visualizer.update_controller_visualization(controller_transform)
            if controller_transform is not None:
                visualizer.update_controller_status_display(
                    controller_transform[:3, 3], connected=True
                )
            else:
                visualizer.update_controller_status_display(None, connected=False)

            # Grip button logic (dead man's switch)
            grip_threshold = visualizer.get_grip_threshold()
            grip_active = grip_value >= grip_threshold

            # Rising edge - grip just pressed
            if (
                grip_active
                and not prev_grip_active
                and controller_transform is not None
            ):
                # Start teleop control - capture initial poses
                controller_initial_transform = controller_transform.copy()

                # Capture initial robot end-effector pose
                current_end_effector_pose = ik_solver.get_current_end_effector_pose()
                position = current_end_effector_pose[:3, 3]
                orientation = current_end_effector_pose[:3, :3]
                robot_initial_transform = np.eye(4)
                robot_initial_transform[:3, :3] = orientation
                robot_initial_transform[:3, 3] = position

                teleop_active = True
                print("✓ Teleop control activated")
                print(
                    f"  Controller initial position: {controller_initial_transform[:3, 3]}"
                )
                print(f"  Robot initial position: {robot_initial_transform[:3, 3]}")

            # Falling edge - grip just released
            elif not grip_active and prev_grip_active:
                # Stop teleop control
                teleop_active = False
                controller_initial_transform = None
                robot_initial_transform = None
                print("✗ Teleop control deactivated")

            prev_grip_active = grip_active

            # Handle homing button (visualization only)
            if visualizer.is_go_home_pressed():
                visualizer.reset_go_home_button()
                print("🏠 GUI: Moving to home position (visualization)...")
                visualizer.update_homing_status("Homing: In progress (GUI)")
                try:
                    # Set IK solver to home configuration
                    ik_solver.set_configuration(np.array(NEUTRAL_JOINT_ANGLES))
                    # Update target transform to match home position
                    current_end_effector_pose = (
                        ik_solver.get_current_end_effector_pose()
                    )
                    position = current_end_effector_pose[:3, 3]
                    orientation = current_end_effector_pose[:3, :3]
                    home_target_transform = np.eye(4)
                    home_target_transform[:3, :3] = orientation
                    home_target_transform[:3, 3] = position
                    visualizer.update_homing_status("Homing: Complete")
                    print("✓ IK solver moved to home position")
                except Exception as e:
                    print(f"✗ Failed to move to home: {e}")
                    visualizer.update_homing_status("Homing: Failed")

            # Update teleop status display
            visualizer.update_teleop_status(teleop_active)

            # Calculate target transform if teleop is active
            target_transform = None
            if (
                teleop_active
                and controller_transform is not None
                and controller_initial_transform is not None
                and robot_initial_transform is not None
            ):
                # Calculate delta transform in controller space as translation and rotation matrix
                delta_position = (
                    controller_transform[:3, 3] - controller_initial_transform[:3, 3]
                )
                delta_orientation = (
                    controller_transform[:3, :3]
                    @ controller_initial_transform[:3, :3].T
                )

                target_transform = np.eye(4)
                target_transform[:3, 3] = (
                    robot_initial_transform[:3, 3] + delta_position
                )
                # Apply the orientation delta to the initial robot orientation
                target_transform[:3, :3] = (
                    delta_orientation @ robot_initial_transform[:3, :3]
                )

            # Solve IK
            if target_transform is not None:
                ik_solver.set_target_pose(
                    target_transform[:3, 3], target_transform[:3, :3]
                )

            success = ik_solver.solve_ik()

            if success:
                # Update target/goal visualization
                if target_transform is not None:
                    visualizer.update_target_visualization(target_transform)
                else:
                    # Update target to current end effector pose when no target
                    current_end_effector_pose = (
                        ik_solver.get_current_end_effector_pose()
                    )
                    position = current_end_effector_pose[:3, 3]
                    orientation = current_end_effector_pose[:3, :3]
                    current_target_transform = np.eye(4)
                    current_target_transform[:3, :3] = orientation
                    current_target_transform[:3, 3] = position
                    visualizer.update_target_visualization(current_target_transform)

                # Update timing
                stats = ik_solver.get_statistics()
                visualizer.update_timing(stats["last_solve_time_ms"])

                # Get joint configuration (IK solution)
                joint_config = ik_solver.get_current_configuration()

                # Update robot visualization
                visualizer.update_robot_pose(joint_config)
                visualizer.update_joint_angles_display(joint_config)

            time.sleep(1 / IK_SOLVER_RATE)

    except KeyboardInterrupt:
        print("\n\n👋 Interrupt received - shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup (outside try/except so it always runs)
        print("\n🧹 Cleaning up...")
        quest_reader.stop()
        visualizer.stop()
        print("✓ Cleanup completed")


if __name__ == "__main__":
    main()
