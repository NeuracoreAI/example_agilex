#!/usr/bin/env python3
"""Piper Robot Teleoperation Visualization with Meta Quest Controller.

This demo visualizes Pink IK control with Meta Quest controller input.
- NO REAL ROBOT CONTROL - visualization only!
- Uses right hand controller grip as dead man's button
- Uses ROS pointer frame for natural pointing control
- Applies relative transformations accounting for different coordinate frames
- Shows what commands WOULD be sent to robot
"""

import sys
from pathlib import Path

# Add parent directory to path to import pink_ik_solver
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add meta_quest_reader to path
sys.path.insert(0, str(Path(__file__).parent.parent / "meta_quest_reader"))

import time

import numpy as np
import viser
import yourdfpy
from configs import (
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
    SMOOTHING_ALPHA,
    SOLVER_DAMPING_VALUE,
    SOLVER_NAME,
    URDF_PATH,
)
from scipy.spatial.transform import Rotation, Slerp
from viser.extras import ViserUrdf

from meta_quest_reader.reader import MetaQuestReader
from pink_ik_solver import PinkIKSolver


def main() -> None:
    """Main function for Piper teleoperation visualization demo."""
    print("=" * 60)
    print("PIPER ROBOT TELEOPERATION VISUALIZATION (NO REAL ROBOT)")
    print("=" * 60)

    # Initialize Meta Quest reader
    print("\n🎮 Initializing Meta Quest reader...")
    quest_reader = MetaQuestReader(
        ip_address=None, port=5555, print_FPS=False, run=True
    )

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
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

    # Load URDF for visualization
    urdf = yourdfpy.URDF.load(URDF_PATH)

    # Robot visualization - shows IK solution (what would be commanded)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # Create transform control for controller visualization
    controller_handle = server.scene.add_transform_controls(
        "/controller",
        scale=0.15,
        position=(0, 0, 0),
        wxyz=(1, 0, 0, 0),
    )

    # Create coordinate frame for target/goal visualization
    target_frame_handle = server.scene.add_frame(
        "/target_goal", axes_length=0.1, axes_radius=0.003
    )

    # Add GUI controls
    timing_handle = server.gui.add_number("IK Solve Time (ms)", 0.001, disabled=True)
    grip_value_handle = server.gui.add_number("Grip Value", 0.0, disabled=True)
    teleop_status_handle = server.gui.add_text("Teleop Status", "Inactive")
    controller_status_handle = server.gui.add_text("Controller Status", "Waiting...")
    joint_angles_handle = server.gui.add_text(
        "Joint Angles (IK Solution)", "Waiting..."
    )

    # Homing controls
    homing_status_handle = server.gui.add_text("Homing Status", "Idle")
    go_home_button = server.gui.add_button("Go Home")

    # Smoothing control
    smoothing_alpha_handle = server.gui.add_number(
        "Smoothing Factor", SMOOTHING_ALPHA, min=0.0, max=1.0, step=0.01
    )

    # Pink parameters
    server.gui.add_text("", "--- IK Parameters ---")
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

    grip_threshold_handle = server.gui.add_number(
        "Grip Threshold", GRIP_THRESHOLD, min=0.0, max=1.0, step=0.05
    )

    # Register Button B callback for home position (visualization only)
    def on_button_b_pressed() -> None:
        """Handle Button B press to move robot to home position (visualization)."""
        print("🏠 Button B pressed - Moving to home position (visualization)...")
        homing_status_handle.value = "Homing: In progress (Button B)"
        # Set IK solver to home configuration
        try:
            ik_solver.set_configuration(np.array(NEUTRAL_JOINT_ANGLES))
            # Update target transform to match home position
            position, orientation = ik_solver.get_current_end_effector_pose()
            home_target_transform = np.eye(4)
            home_target_transform[:3, :3] = orientation
            home_target_transform[:3, 3] = position
            homing_status_handle.value = "Homing: Complete (Button B)"
            print("✓ IK solver moved to home position")
        except Exception as e:
            print(f"✗ Failed to move to home: {e}")
            homing_status_handle.value = "Homing: Failed"

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

    # Smoothing state
    smoothing_alpha = SMOOTHING_ALPHA
    controller_transform_smoothed = None

    # Timing display state
    ema_timing = 0.001  # Exponential moving average for timing display

    try:
        while True:
            # Update Meta Quest data
            quest_reader.update()

            # Update Pink parameters
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
                posture_cost_vector=posture_cost_vector,
            )

            # Get controller data
            grip_value = quest_reader.get_grip_value("right")
            grip_value_handle.value = grip_value

            controller_transform_raw = quest_reader.get_hand_controller_transform_ros(
                hand="right"
            )

            # Apply smoothing to controller transform
            if controller_transform_raw is not None:
                if controller_transform_smoothed is None:
                    # First frame - no smoothing
                    controller_transform_smoothed = controller_transform_raw.copy()
                else:
                    # Apply EMA smoothing to transform
                    smoothing_alpha = smoothing_alpha_handle.value

                    # Smooth position using EMA
                    prev_pos = controller_transform_smoothed[:3, 3]
                    new_pos = controller_transform_raw[:3, 3]
                    smoothed_pos = (
                        1 - smoothing_alpha
                    ) * prev_pos + smoothing_alpha * new_pos

                    # Smooth orientation using SLERP (quaternion spherical interpolation)
                    prev_rot = Rotation.from_matrix(
                        controller_transform_smoothed[:3, :3]
                    )
                    new_rot = Rotation.from_matrix(controller_transform_raw[:3, :3])

                    # SLERP interpolation using SciPy's Slerp class
                    key_times = [0.0, 1.0]
                    key_rots = Rotation.concatenate([prev_rot, new_rot])
                    slerp = Slerp(key_times, key_rots)
                    smoothed_rot = slerp([smoothing_alpha])[0]

                    # Build smoothed transform
                    controller_transform_smoothed = np.eye(4)
                    controller_transform_smoothed[:3, 3] = smoothed_pos
                    controller_transform_smoothed[:3, :3] = smoothed_rot.as_matrix()
            else:
                controller_transform_smoothed = None

            controller_transform = controller_transform_smoothed

            # Update controller visualization if available
            if controller_transform is not None:
                controller_pos = controller_transform[:3, 3]
                controller_rot = Rotation.from_matrix(controller_transform[:3, :3])
                controller_quat_xyzw = controller_rot.as_quat()
                controller_quat_wxyz = [
                    controller_quat_xyzw[3],
                    controller_quat_xyzw[0],
                    controller_quat_xyzw[1],
                    controller_quat_xyzw[2],
                ]

                # set controller visualization to current controller position and orientation
                controller_handle.position = tuple(controller_pos)
                controller_handle.wxyz = tuple(controller_quat_wxyz)

                # update controller status display
                controller_status_str = "Controller Status:\n"
                controller_status_str += f"  Position: [{controller_pos[0]:.3f}, {controller_pos[1]:.3f}, {controller_pos[2]:.3f}]\n"
                controller_status_str += "  Connected: ✓\n"
                controller_status_handle.value = controller_status_str
            else:
                controller_status_handle.value = "Controller Status:\n  Connected: ✗"

            # Grip button logic (dead man's switch)
            grip_threshold = grip_threshold_handle.value
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
                position, orientation = ik_solver.get_current_end_effector_pose()
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
            if go_home_button.value:
                go_home_button.value = False
                print("🏠 GUI: Moving to home position (visualization)...")
                homing_status_handle.value = "Homing: In progress (GUI)"
                try:
                    # Set IK solver to home configuration
                    ik_solver.set_configuration(np.array(NEUTRAL_JOINT_ANGLES))
                    # Update target transform to match home position
                    position, orientation = ik_solver.get_current_end_effector_pose()
                    home_target_transform = np.eye(4)
                    home_target_transform[:3, :3] = orientation
                    home_target_transform[:3, 3] = position
                    homing_status_handle.value = "Homing: Complete"
                    print("✓ IK solver moved to home position")
                except Exception as e:
                    print(f"✗ Failed to move to home: {e}")
                    homing_status_handle.value = "Homing: Failed"

            # Update teleop status display - simple Active/Inactive
            if teleop_active:
                teleop_status_handle.value = "Teleop Status: Active"
            else:
                teleop_status_handle.value = "Teleop Status: Inactive"

            # Calculate target transform if teleop is active
            target_transform = None
            if (
                teleop_active
                and controller_transform is not None
                and controller_initial_transform is not None
                and robot_initial_transform is not None
            ):
                # # OPTION1: Calculate delta transform in controller space in 4x4 homogeneous transformation matrix
                # # T_delta = T_initial^-1 * T_current
                # controller_initial_inv = np.linalg.inv(controller_initial_transform)
                # T_delta = controller_initial_inv @ controller_transform
                # # Apply delta to robot initial pose
                # # T_robot_target = T_robot_initial * T_delta
                # T_robot_target = robot_initial_transform @ T_delta

                # # OPTION2: Calculate delta transform in controller space as translation and rotation euler angles
                # delta_position = (
                #     controller_transform[:3, 3] - controller_initial_transform[:3, 3]
                # )
                # delta_orientation = Rotation.from_matrix(
                #     controller_transform[:3, :3]
                # ).as_euler("xyz") - Rotation.from_matrix(
                #     controller_initial_transform[:3, :3]
                # ).as_euler(
                #     "xyz"
                # )
                # T_robot_target = np.eye(4)
                # T_robot_target[:3, 3] = robot_initial_transform[:3, 3] + delta_position
                # T_robot_target[:3, :3] = Rotation.from_euler(
                #     "xyz", delta_orientation
                # ).as_matrix()

                # OPTION3: Calculate delta transform in controller space as translation and rotation matrix
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
                    target_pos = target_transform[:3, 3]
                    target_rot = Rotation.from_matrix(target_transform[:3, :3])
                    target_quat_xyzw = target_rot.as_quat()
                    target_quat_wxyz = [
                        target_quat_xyzw[3],
                        target_quat_xyzw[0],
                        target_quat_xyzw[1],
                        target_quat_xyzw[2],
                    ]

                    target_frame_handle.position = tuple(target_pos)
                    target_frame_handle.wxyz = tuple(target_quat_wxyz)
                else:
                    # Update target to current end effector pose when no target
                    position, orientation = ik_solver.get_current_end_effector_pose()
                    current_target_transform = np.eye(4)
                    current_target_transform[:3, :3] = orientation
                    current_target_transform[:3, 3] = position
                    target_pos = current_target_transform[:3, 3]
                    target_rot = Rotation.from_matrix(current_target_transform[:3, :3])
                    target_quat_xyzw = target_rot.as_quat()
                    target_quat_wxyz = [
                        target_quat_xyzw[3],
                        target_quat_xyzw[0],
                        target_quat_xyzw[1],
                        target_quat_xyzw[2],
                    ]
                    target_frame_handle.position = tuple(target_pos)
                    target_frame_handle.wxyz = tuple(target_quat_wxyz)
                # Update timing
                stats = ik_solver.get_statistics()
                solve_time_ms = stats["last_solve_time_ms"]
                # Update timing display with exponential moving average
                ema_timing = 0.99 * ema_timing + 0.01 * solve_time_ms
                timing_handle.value = ema_timing

                # Get joint configuration (IK solution)
                joint_config = ik_solver.get_current_configuration()

                # Update robot visualization
                urdf_vis.update_cfg(joint_config)

                # Update joint angles display
                joint_angles_str = "Joint Angles (IK Solution):\n"
                joint_angles_deg = np.degrees(joint_config)
                for i in range(6):
                    angle_rad = joint_config[i]
                    angle_deg = joint_angles_deg[i]
                    joint_angles_str += (
                        f"  J{i+1}: {angle_rad:.3f} rad ({angle_deg:.1f}°)\n"
                    )
                joint_angles_handle.value = joint_angles_str

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
        server.stop()
        print("✓ Cleanup completed")


if __name__ == "__main__":
    main()
