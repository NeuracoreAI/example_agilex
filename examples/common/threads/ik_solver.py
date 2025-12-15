"""IK solver thread - solves IK and updates state."""

import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Add parent directory to path to import pink_ik_solver
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.configs import IK_SOLVER_RATE, ROTATION_SCALE, TRANSLATION_SCALE
from common.data_manager import DataManager, RobotActivityState
from common.utils import scale_and_add_delta_transform

from pink_ik_solver import PinkIKSolver


def ik_solver_thread(data_manager: DataManager, ik_solver: PinkIKSolver) -> None:
    """IK solver thread - solves IK and updates state."""
    print("🧮 IK solver thread started")

    dt: float = 1.0 / IK_SOLVER_RATE
    DIVERGENCE_TOLERANCE = 0.1

    try:
        while not data_manager.is_shutdown_requested():
            iteration_start: float = time.time()

            # Get current robot joint angles from state
            current_joint_angles = data_manager.get_current_joint_angles()
            current_ik_joint_angles = np.degrees(ik_solver.get_current_configuration())

            # Sync IK solver with actual joint angles if close enough
            if (
                current_joint_angles is not None
                and current_ik_joint_angles is not None
                and np.all(
                    np.abs(current_joint_angles - current_ik_joint_angles)
                    <= DIVERGENCE_TOLERANCE
                )
            ):
                ik_solver.set_configuration_no_task_update(
                    np.radians(current_joint_angles)
                )

            # Get current end effector pose from IK solver and set in state
            ik_ee_pose = ik_solver.get_current_end_effector_pose()
            data_manager.set_current_end_effector_pose(ik_ee_pose)

            # Get current state
            robot_activity_state = data_manager.get_robot_activity_state()
            controller_transform, _, _ = data_manager.get_controller_data()
            teleop_active = data_manager.get_teleop_active()

            # Skip teleop-based IK if in POLICY_CONTROLLED state
            if robot_activity_state == RobotActivityState.POLICY_CONTROLLED:
                if current_joint_angles is not None:
                    ik_solver.set_configuration(np.radians(current_joint_angles))
                    current_end_effector_pose = (
                        data_manager.get_current_end_effector_pose()
                    )
                    data_manager.set_target_pose(current_end_effector_pose)
                    data_manager.set_target_joint_angles(current_joint_angles)
                    data_manager.set_ik_solve_time_ms(0.0)
                    data_manager.set_ik_success(True)

            elif teleop_active and controller_transform is not None:
                controller_initial, robot_initial = (
                    data_manager.get_initial_robot_controller_transforms()
                )
                if controller_initial is None or robot_initial is None:
                    elapsed = time.time() - iteration_start
                    sleep_time = dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue

                # Calculate delta transform
                delta_position = controller_transform[:3, 3] - controller_initial[:3, 3]
                delta_orientation = (
                    controller_transform[:3, :3] @ controller_initial[:3, :3].T
                )

                T_robot_target = scale_and_add_delta_transform(
                    delta_position,
                    delta_orientation,
                    TRANSLATION_SCALE,
                    ROTATION_SCALE,
                    robot_initial,
                )

                ik_solver.set_target_pose(T_robot_target[:3, 3], T_robot_target[:3, :3])
                data_manager.set_target_pose(T_robot_target)

                # Solve IK
                success = ik_solver.solve_ik()

                if success:
                    joint_config = np.degrees(ik_solver.get_current_configuration())
                    stats = ik_solver.get_statistics()
                    solve_time_ms = stats["last_solve_time_ms"]

                    data_manager.set_target_joint_angles(joint_config)
                    data_manager.set_ik_solve_time_ms(solve_time_ms)
                    data_manager.set_ik_success(success)
                else:
                    data_manager.set_ik_solve_time_ms(0.0)
                    data_manager.set_ik_success(False)

            else:  # robot is HOMING or DISABLED
                if current_joint_angles is not None:
                    ik_solver.set_configuration(np.radians(current_joint_angles))
                    current_end_effector_pose = (
                        data_manager.get_current_end_effector_pose()
                    )
                    data_manager.set_target_pose(current_end_effector_pose)
                    data_manager.set_target_joint_angles(current_joint_angles)
                    data_manager.set_ik_solve_time_ms(0.0)
                    data_manager.set_ik_success(True)

            # Sleep to maintain loop rate
            elapsed = time.time() - iteration_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"❌ IK solver thread error: {e}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        print("🧮 IK solver thread stopped")
