#!/usr/bin/env python3
"""Piper Robot Pink IK Demo.

This demo uses Pink (Python inverse kinematics based on Pinocchio) to control
the Piper robot with smooth differential inverse kinematics.
"""

import sys
from pathlib import Path

# Add parent directory to path to import pink_ik_solver
sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import numpy as np
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
    SOLVER_DAMPING_VALUE,
    SOLVER_NAME,
    URDF_PATH,
)
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

from pink_ik_solver import PinkIKSolver


def main() -> None:
    """Main function for Piper robot Pink IK demo."""
    print("=" * 60)
    print("PIPER ROBOT PINK IK DEMO")
    print("=" * 60)

    print(f"Using solver: {SOLVER_NAME}")

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

    print("\n🚀 Starting Pink IK demo...")
    print("   Move the red transform controls to see smooth robot motion!")
    print("   Use sliders to adjust task weights and parameters")
    print("   Press Ctrl+C to exit.")
    print("\n🌐 Open browser: http://localhost:8080")

    try:
        while True:
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
                for i, angle in enumerate(joint_config):
                    joint_angles_str += f"  Joint {i+1}: {angle:.3f}\n"
                joint_angles_handle.value = joint_angles_str
            else:
                joint_angles_handle.value = "IK Failed: Check console for details"

            time.sleep(1 / ik_solver_rate_handle.value)  # Real-time control

    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user.")
    except Exception as e:
        print(f"❌ Demo error: {e}")


if __name__ == "__main__":
    main()
