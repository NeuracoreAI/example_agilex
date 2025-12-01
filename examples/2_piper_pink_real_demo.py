#!/usr/bin/env python3
"""Piper Robot Pink IK Demo with Real Robot Control.

This demo uses Pink to control a Piper robot through Viser on a browser.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add parent directory to path to import pink_ik_solver and piper_controller
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import (
    DAMPING_COST,
    FRAME_TASK_GAIN,
    GRIPPER_FRAME_NAME,
    IK_SOLVER_RATE,
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
from common.robot_visualizer import RobotVisualizer
from scipy.spatial.transform import Rotation

from pink_ik_solver import PinkIKSolver
from piper_controller import PiperController


def main() -> None:
    """Main function for Piper robot Pink IK demo with real robot control."""
    print("=" * 60)
    print("PIPER ROBOT PINK IK DEMO WITH REAL ROBOT CONTROL")
    print("=" * 60)

    print(f"Using solver: {SOLVER_NAME}")

    # Initialize robot controller in joint space mode
    print("\n🤖 Initializing Piper robot controller...")
    robot_controller = PiperController(
        can_interface="can0",
        robot_rate=ROBOT_RATE,
        control_mode=PiperController.ControlMode.JOINT_SPACE,
        debug_mode=False,
    )

    # Get current robot joint angles and use them as initial configuration
    print("\n📊 Getting current robot joint angles...")
    current_joint_angles = robot_controller.get_current_joint_angles()
    if current_joint_angles is not None:
        initial_configuration = np.radians(current_joint_angles)
    else:
        print("Could not get current robot joints, will use neutral configuration")
        initial_configuration = np.array(NEUTRAL_JOINT_ANGLES)

    print(f"\n📊 Initial configuration (radians): {initial_configuration}")

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
        initial_configuration=initial_configuration,
        posture_cost_vector=np.array(POSTURE_COST_VECTOR),
    )

    # Get initial end effector pose
    current_end_effector_pose = ik_solver.get_current_end_effector_pose()

    # Set up visualizer
    print("\n🖥️  Starting visualization...")
    visualizer = RobotVisualizer(urdf_path=URDF_PATH)
    visualizer.add_basic_controls()
    visualizer.add_robot_status_controls()
    visualizer.add_safety_status_controls()
    visualizer.add_pink_parameter_controls(
        position_cost=POSITION_COST,
        orientation_cost=ORIENTATION_COST,
        frame_task_gain=FRAME_TASK_GAIN,
        lm_damping=LM_DAMPING,
        damping_cost=DAMPING_COST,
        solver_damping_value=SOLVER_DAMPING_VALUE,
        posture_cost_vector=POSTURE_COST_VECTOR,
    )
    visualizer.add_ik_solver_rate_control(initial_rate=IK_SOLVER_RATE)
    visualizer.add_robot_control_buttons()
    visualizer.add_homing_controls()
    visualizer.add_ik_target_controls(
        current_end_effector_pose[:3, 3], current_end_effector_pose[:3, :3]
    )

    # Start robot control loop
    print("\n🚀 Starting robot control loop...")
    robot_controller.start_control_loop()

    print("\n🚀 Starting Pink IK demo with real robot control...")
    print("   Move the red transform controls to see smooth robot motion!")
    print("   Use sliders to adjust task weights and parameters")
    print("   Robot control buttons:")
    print("     - Enable Robot: Start sending commands to robot")
    print("     - Disable Robot: Stop sending commands to robot")
    print("     - Emergency Stop: Immediately halt robot")
    print("   Homing controls:")
    print("     - Go Home: Return robot to home position")
    print("   Press Ctrl+C to exit.")
    print("\n🌐 Open browser: http://localhost:8080")

    # Variables for robot control
    robot_enabled = False
    last_joint_command_time: float = 0.0
    joint_command_rate = 0.01  # Send joint commands at 100 Hz

    try:
        while True:
            # Handle GUI button presses
            if visualizer.is_enable_robot_pressed():
                visualizer.reset_enable_robot_button()
                if robot_controller.resume_robot():
                    robot_enabled = True
                    print("✓ Robot enabled")
                else:
                    print("✗ Failed to enable robot")

            if visualizer.is_disable_robot_pressed():
                visualizer.reset_disable_robot_button()
                robot_enabled = False
                robot_controller.graceful_stop()
                print("✓ Robot disabled")

            if visualizer.is_emergency_stop_pressed():
                visualizer.reset_emergency_stop_button()
                robot_enabled = False
                robot_controller.emergency_stop()
                print("🚨 Emergency stop activated!")

            if visualizer.is_go_home_pressed():
                visualizer.reset_go_home_button()

                # Move robot to home position
                robot_controller.move_to_home()

                # Get home pose from robot controller and update GUI transform controls
                home_pose = robot_controller.HOME_POSE
                home_position = np.array(home_pose[:3])  # [x, y, z] in mm
                home_orientation_euler = np.array(
                    home_pose[3:6]
                )  # [rx, ry, rz] in degrees

                # Update GUI transform controls to home position
                home_position_m = home_position / 1000.0  # Convert mm to m for GUI
                home_orientation = Rotation.from_euler(
                    "xyz", home_orientation_euler
                ).as_matrix()
                visualizer.set_ik_target_pose(home_position_m, home_orientation)

                print("🏠 Moving to home position...")

            # Update task parameters from GUI
            pink_params = visualizer.get_pink_parameters()
            ik_solver.update_task_parameters(**pink_params)

            # Get target from GUI
            target_position, target_rotation_matrix = visualizer.get_ik_target_pose()

            # Update IK solver target
            ik_solver.set_target_pose(target_position, target_rotation_matrix)

            # Solve differential IK
            success = ik_solver.solve_ik()

            if success:
                # Update timing
                stats = ik_solver.get_statistics()
                visualizer.update_timing(stats["last_solve_time_ms"])

                # Update robot visualization
                joint_config = ik_solver.get_current_configuration()
                visualizer.update_robot_pose(joint_config)
                visualizer.update_joint_angles_display(joint_config, show_gripper=True)

                # Send joint commands to robot if enabled and rate-limited
                current_time = time.time()
                if (
                    robot_enabled
                    and (current_time - last_joint_command_time) >= joint_command_rate
                ):
                    try:
                        # Convert Pink joint angles (radians) to robot units (degrees)
                        # Only use first 6 joints for robot control (joints 7&8 are gripper joints in URDF)
                        robot_joint_angles = np.degrees(joint_config[:6])

                        # Send to robot controller (PiperController handles joint limit clamping)
                        robot_controller.set_target_joint_angles(
                            robot_joint_angles.tolist()
                        )
                        last_joint_command_time = current_time

                    except Exception as e:
                        print(f"Failed to send joint command to robot: {e}")
                        # Disable robot on error for safety
                        robot_enabled = False

                # Update robot status display
                robot_status = robot_controller.get_robot_status()
                status_str = "Robot Status:\n"
                status_str += f"  Enabled: {robot_status['enabled']}\n"
                status_str += f"  Control Mode: {robot_status['control_mode']}\n"
                if robot_status["current_joint_angles"]:
                    status_str += f"  Current Joints: {[f'{j:.1f}°' for j in robot_status['current_joint_angles']]}\n"
                if robot_status["current_gripper_open_value"]:
                    status_str += f"  Gripper: {robot_status['current_gripper_open_value']:.3f} (normalized)\n"
                visualizer.update_robot_status(status_str)

                # Update safety status display
                safety_str = "Safety Status:\n"
                safety_str += (
                    "  Joint Limits: ✓ Handled by IK solver & robot controller\n"
                )
                safety_str += (
                    f"  Robot Enabled: {'✓ YES' if robot_enabled else '✗ NO'}\n"
                )
                safety_str += f"  Control Active: {'✓ YES' if robot_status['enabled'] else '✗ NO'}\n"
                visualizer.update_safety_status(safety_str)
            else:
                visualizer.set_joint_angles_text("IK Failed: Check console for details")

            time.sleep(1 / visualizer.get_ik_solver_rate())  # Real-time control

    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user.")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        robot_enabled = False
        robot_controller.cleanup()
        visualizer.stop()
        print("✓ Cleanup completed")


if __name__ == "__main__":
    main()
