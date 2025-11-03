#!/usr/bin/env python3
"""Piper Robot Pink IK Demo with Real Robot Control.

This demo uses Pink (Python inverse kinematics based on Pinocchio) to control
the real Piper robot with smooth differential inverse kinematics.

Features:
- Real-time Pink IK solving using PinkIKSolver class
- Joint space control of physical robot
- Safety features (emergency stop, joint limits)
- Visual feedback with Viser
- Real robot status monitoring
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add parent directory to path to import pink_ik_solver and piper_controller
sys.path.insert(0, str(Path(__file__).parent.parent))

import viser
import yourdfpy
from configs import (
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
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

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
    initial_configuration = np.radians(robot_controller.get_current_joint_angles())

    if initial_configuration is None:
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
    current_position, current_orientation = ik_solver.get_current_end_effector_pose()

    # Set up visualizer
    print("\n🖥️  Starting visualization...")
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

    # Load URDF for visualization
    urdf = yourdfpy.URDF.load(URDF_PATH)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # Create interactive controller for IK target
    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target",
        scale=0.2,
        position=tuple(current_position),  # Start position from current pose
        wxyz=tuple(
            Rotation.from_matrix(current_orientation).as_quat()[[3, 0, 1, 2]]
        ),  # wxyz from current orientation
    )

    # Add GUI controls
    timing_handle = server.gui.add_number("IK Solve Time (ms)", 0.001, disabled=True)
    joint_angles_handle = server.gui.add_text(
        "Joint Angles", "Waiting for IK solution..."
    )
    robot_status_handle = server.gui.add_text("Robot Status", "Initializing...")
    safety_status_handle = server.gui.add_text("Safety Status", "Monitoring...")

    # Pink-specific controls
    position_weight_handle = server.gui.add_number(
        "Position Weight", POSITION_COST, min=0.0, max=10.0, step=0.1
    )
    orientation_weight_handle = server.gui.add_number(
        "Orientation Weight", ORIENTATION_COST, min=0.0, max=1.0, step=0.01
    )
    frame_task_gain_handle = server.gui.add_number(
        "Frame Task Gain", FRAME_TASK_GAIN, min=0.0, max=10.0, step=0.1
    )
    lm_damping_handle = server.gui.add_number(
        "LM Damping", LM_DAMPING, min=0.0, max=5.0, step=0.01
    )
    damping_weight_handle = server.gui.add_number(
        "Damping Weight", DAMPING_COST, min=0.0, max=1.0, step=0.01
    )
    solver_damping_value_handle = server.gui.add_number(
        "Solver Damping Value", SOLVER_DAMPING_VALUE, min=0.0, max=1.0, step=0.0001
    )

    # Posture cost controls (one per joint)
    posture_cost_handles = []
    for i in range(len(POSTURE_COST_VECTOR)):
        handle = server.gui.add_number(
            f"Posture Cost J{i+1}", POSTURE_COST_VECTOR[i], min=0.0, max=1.0, step=0.01
        )
        posture_cost_handles.append(handle)

    ik_solver_rate_handle = server.gui.add_number(
        "IK Solver Rate (Hz)", IK_SOLVER_RATE, min=100.0, max=1000.0, step=10.0
    )

    # Robot control controls
    enable_robot_handle = server.gui.add_button("Enable Robot")
    disable_robot_handle = server.gui.add_button("Disable Robot")
    emergency_stop_handle = server.gui.add_button("Emergency Stop")
    move_home_handle = server.gui.add_button("Move to Home")

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
    print("     - Move to Home: Return robot to home position")
    print("   Press Ctrl+C to exit.")
    print("\n🌐 Open browser: http://localhost:8080")

    # Variables for robot control
    robot_enabled = False
    last_joint_command_time: float = 0.0
    joint_command_rate = 0.01  # Send joint commands at 100 Hz

    try:
        while True:
            # Handle GUI button presses
            if enable_robot_handle.value:
                enable_robot_handle.value = False  # Reset button state
                if robot_controller.resume_robot():
                    robot_enabled = True
                    print("✓ Robot enabled")
                else:
                    print("✗ Failed to enable robot")

            if disable_robot_handle.value:
                disable_robot_handle.value = False  # Reset button state
                robot_enabled = False
                robot_controller.graceful_stop()
                print("✓ Robot disabled")

            if emergency_stop_handle.value:
                emergency_stop_handle.value = False  # Reset button state
                robot_enabled = False
                robot_controller.emergency_stop()
                print("🚨 Emergency stop activated!")

            if move_home_handle.value:
                move_home_handle.value = False  # Reset button state

                # Move robot to home position
                robot_controller.move_to_home()

                # Get home pose from robot controller and update GUI transform controls
                home_pose = robot_controller.HOME_POSE
                home_position = np.array(home_pose[:3])  # [x, y, z] in mm
                home_orientation_euler = np.array(
                    home_pose[3:6]
                )  # [rx, ry, rz] in degrees

                # Convert euler angles to rotation matrix, then to quaternion
                home_rotation_matrix = Rotation.from_euler(
                    "xyz", home_orientation_euler, degrees=True
                ).as_matrix()
                home_quat_xyzw = Rotation.from_matrix(home_rotation_matrix).as_quat()
                home_quat_wxyz = np.array(
                    [
                        home_quat_xyzw[3],
                        home_quat_xyzw[0],
                        home_quat_xyzw[1],
                        home_quat_xyzw[2],
                    ]
                )

                # Update GUI transform controls to home position
                ik_target_handle.position = tuple(
                    home_position / 1000.0
                )  # Convert mm to m for GUI
                ik_target_handle.wxyz = tuple(home_quat_wxyz)

                print("🏠 Moving to home position...")

            # Update task parameters from GUI
            # Collect posture cost values from GUI handles
            posture_cost_vector = np.array(
                [handle.value for handle in posture_cost_handles]
            )
            ik_solver.update_task_parameters(
                position_cost=position_weight_handle.value,
                orientation_cost=orientation_weight_handle.value,
                frame_task_gain=frame_task_gain_handle.value,
                lm_damping=lm_damping_handle.value,
                damping_cost=damping_weight_handle.value,
                solver_damping_value=solver_damping_value_handle.value,
                integration_time_step=1 / ik_solver_rate_handle.value,
                posture_cost_vector=posture_cost_vector,
            )

            # Get target from GUI
            target_position = np.array(ik_target_handle.position)
            target_wxyz = np.array(ik_target_handle.wxyz)

            # Convert wxyz quaternion to rotation matrix for Pinocchio
            target_rotation = Rotation.from_quat(
                [target_wxyz[1], target_wxyz[2], target_wxyz[3], target_wxyz[0]]
            )  # wxyz to xyzw
            target_rotation_matrix = target_rotation.as_matrix()

            # Update IK solver target
            ik_solver.set_target_pose(target_position, target_rotation_matrix)

            # Solve differential IK
            success = ik_solver.solve_ik()

            if success:
                # Update timing handle
                stats = ik_solver.get_statistics()
                timing_handle.value = (
                    0.99 * timing_handle.value + 0.01 * stats["last_solve_time_ms"]
                )

                # Update robot visualization
                joint_config = ik_solver.get_current_configuration()
                urdf_vis.update_cfg(joint_config)

                # Update joint angles display
                joint_angles_str = "Joint Angles (rad):\n"
                joint_angles_deg = np.degrees(joint_config)
                for i, (angle_rad, angle_deg) in enumerate(
                    zip(joint_config, joint_angles_deg)
                ):
                    joint_type = "Robot" if i < 6 else "Gripper"
                    joint_angles_str += f"  Joint {i+1} ({joint_type}): {angle_rad:.3f} rad ({angle_deg:.1f}°)\n"
                joint_angles_handle.value = joint_angles_str

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
                    status_str += f"  Gripper: {robot_status['current_gripper_open_value']:.1f}°\n"
                robot_status_handle.value = status_str

                # Update safety status display
                safety_str = "Safety Status:\n"
                safety_str += (
                    "  Joint Limits: ✓ Handled by IK solver & robot controller\n"
                )
                safety_str += (
                    f"  Robot Enabled: {'✓ YES' if robot_enabled else '✗ NO'}\n"
                )
                safety_str += f"  Control Active: {'✓ YES' if robot_status['enabled'] else '✗ NO'}\n"
                safety_status_handle.value = safety_str
            else:
                joint_angles_handle.value = "IK Failed: Check console for details"

            time.sleep(1 / ik_solver_rate_handle.value)  # Real-time control

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
        print("✓ Cleanup completed")


if __name__ == "__main__":
    main()
