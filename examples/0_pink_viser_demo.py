#!/usr/bin/env python3
"""Piper Robot Pink IK Demo on Viser.

This demo uses Pink to control a Piper robot on Viser.
"""

import sys
from pathlib import Path

# Add parent directory to path to import pink_ik_solver
sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import numpy as np
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
    SOLVER_DAMPING_VALUE,
    SOLVER_NAME,
    URDF_PATH,
)
from common.robot_visualizer import RobotVisualizer

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
    current_end_effector_pose = ik_solver.get_current_end_effector_pose()
    current_position = current_end_effector_pose[:3, 3]
    current_orientation = current_end_effector_pose[:3, :3]

    # Set up visualizer
    print("\n🖥️  Starting visualization...")
    visualizer = RobotVisualizer(urdf_path=URDF_PATH)
    visualizer.add_basic_controls()
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
    visualizer.add_ik_target_controls(current_position, current_orientation)

    print("\n🚀 Starting Pink IK demo...")
    print("   Move the red transform controls to see smooth robot motion!")
    print("   Use sliders to adjust task weights and parameters")
    print("   Press Ctrl+C to exit.")
    print("\n🌐 Open browser: http://localhost:8080")

    try:
        while True:
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
                visualizer.update_joint_angles_display(joint_config)
            else:
                visualizer.set_joint_angles_text("IK Failed: Check console for details")

            time.sleep(1 / visualizer.get_ik_solver_rate())  # Real-time control

    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user.")
    except Exception as e:
        print(f"❌ Demo error: {e}")


if __name__ == "__main__":
    main()
