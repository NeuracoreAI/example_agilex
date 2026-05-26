import threading
import numpy as np
from typing import Tuple, List, Optional

from common.configs import (
    CONTROLLER_BETA, CONTROLLER_D_CUTOFF, CONTROLLER_MIN_CUTOFF,
    ROBOT_RATE, NEUTRAL_JOINT_ANGLES, NEUTRAL_END_EFFECTOR_POSE,
    URDF_PATH, GRIPPER_FRAME_NAME, SOLVER_NAME, POSITION_COST,
    ORIENTATION_COST, FRAME_TASK_GAIN, LM_DAMPING, DAMPING_COST,
    SOLVER_DAMPING_VALUE, IK_SOLVER_RATE, POSTURE_COST_VECTOR,
    TRANSLATION_SCALE, ROTATION_SCALE
)
from common.data_manager import DataManager
from piper_controller import PiperController
from pink_ik_solver import PinkIKSolver
from common.threads.joint_state import joint_state_thread
from common.threads.ik_solver import ik_solver_thread
from common.threads.realsense_camera import camera_thread

def bootstrap_robot_system(
    start_ik: bool = True,
    start_camera: bool = True
) -> Tuple[DataManager, PiperController, Optional[PinkIKSolver], List[threading.Thread]]:
    """Initializes DataManager, PiperController, IK, and base background threads."""
    
    # 1. Initialize Data Manager
    data_manager = DataManager()
    data_manager.set_controller_filter_params(
        CONTROLLER_MIN_CUTOFF, CONTROLLER_BETA, CONTROLLER_D_CUTOFF
    )
    data_manager.set_teleop_scaling(TRANSLATION_SCALE, ROTATION_SCALE)

    # 2. Initialize Robot Controller
    print("\n🤖 Initializing Piper robot controller...")
    robot_controller = PiperController(
        can_interface="can0",
        robot_rate=ROBOT_RATE,
        control_mode=PiperController.ControlMode.JOINT_SPACE,
        neutral_joint_angles=NEUTRAL_JOINT_ANGLES,
        neutral_end_effector_pose=NEUTRAL_END_EFFECTOR_POSE,
        enable_joint_angle_limits=False,
        debug_mode=False,
    )
    robot_controller.start_control_loop()

    # 3. Start Threads
    active_threads = []
    
    print("\n📊 Starting joint state thread...")
    js_thread = threading.Thread(
        target=joint_state_thread, args=(data_manager, robot_controller), daemon=True
    )
    js_thread.start()
    active_threads.append(js_thread)

    ik_solver = None
    if start_ik:
        print("\n🔧 Creating Pink IK solver...")
        current_angles = data_manager.get_current_joint_angles()
        init_angles = np.radians(current_angles) if current_angles is not None else np.radians(NEUTRAL_JOINT_ANGLES)
        
        ik_solver = PinkIKSolver(
            urdf_path=URDF_PATH, end_effector_frame=GRIPPER_FRAME_NAME,
            solver_name=SOLVER_NAME, position_cost=POSITION_COST,
            orientation_cost=ORIENTATION_COST, frame_task_gain=FRAME_TASK_GAIN,
            lm_damping=LM_DAMPING, damping_cost=DAMPING_COST,
            solver_damping_value=SOLVER_DAMPING_VALUE,
            integration_time_step=1 / IK_SOLVER_RATE,
            initial_configuration=init_angles,
            posture_cost_vector=np.array(POSTURE_COST_VECTOR),
        )
        print("\n🧮 Starting IK solver thread...")
        ik_thread = threading.Thread(
            target=ik_solver_thread, args=(data_manager, ik_solver), daemon=True
        )
        ik_thread.start()
        active_threads.append(ik_thread)

    if start_camera:
        print("\n📷 Starting camera thread...")
        cam_thread = threading.Thread(
            target=camera_thread, args=(data_manager,), daemon=True
        )
        cam_thread.start()
        active_threads.append(cam_thread)

    return data_manager, robot_controller, ik_solver, active_threads