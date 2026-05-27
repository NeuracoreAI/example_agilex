import threading
import numpy as np
from typing import Tuple, List, Optional

from common.configs import (
    ROBOT_RATE, NEUTRAL_JOINT_ANGLES, NEUTRAL_END_EFFECTOR_POSE,
    URDF_PATH, GRIPPER_FRAME_NAME, SOLVER_NAME, IK_SOLVER_RATE
)
from common.data_manager import DataManager
from piper_controller import PiperController
from pink_ik_solver import PinkIKSolver
from common.threads.joint_state import joint_state_thread
from common.threads.ik_solver import ik_solver_thread
from common.threads.realsense_camera import camera_thread

def bootstrap_robot_system(
    config: dict,
    start_ik: bool = True,
    start_camera: bool = True
) -> Tuple[DataManager, PiperController, Optional[PinkIKSolver], List[threading.Thread]]:
    
    # Extract config sections safely
    filt_p = config.get("filter_parameters", {})
    tele_p = config.get("teleop_parameters", {})
    ik_p = config.get("ik_parameters", {})

    # 1. Initialize Data Manager
    data_manager = DataManager()
    data_manager.set_controller_filter_params(
        filt_p.get("controller_min_cutoff", 0.8), 
        filt_p.get("controller_beta", 0.05), 
        filt_p.get("controller_d_cutoff", 0.9)
    )
    data_manager.set_teleop_scaling(
        tele_p.get("translation_scale", 1.5), 
        tele_p.get("rotation_scale", 1.2)
    )

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
    js_thread = threading.Thread(target=joint_state_thread, args=(data_manager, robot_controller), daemon=True)
    js_thread.start()
    active_threads.append(js_thread)

    ik_solver = None
    if start_ik:
        print("\n🔧 Creating Pink IK solver...")
        current_angles = data_manager.get_current_joint_angles()
        init_angles = np.radians(current_angles) if current_angles is not None else np.radians(NEUTRAL_JOINT_ANGLES)
        
        ik_solver = PinkIKSolver(
            urdf_path=URDF_PATH, end_effector_frame=GRIPPER_FRAME_NAME,
            solver_name=SOLVER_NAME, 
            position_cost=ik_p.get("position_cost", 1.0),
            orientation_cost=ik_p.get("orientation_cost", 0.75), 
            frame_task_gain=ik_p.get("frame_task_gain", 0.4),
            lm_damping=ik_p.get("lm_damping", 0.01), 
            damping_cost=ik_p.get("damping_cost", 0.25),
            solver_damping_value=ik_p.get("solver_damping_value", 1e-4),
            integration_time_step=1 / IK_SOLVER_RATE,
            initial_configuration=init_angles,
            posture_cost_vector=np.array(ik_p.get("posture_cost_vector", [0.0, 0.0, 0.0, 0.05, 0.0, 0.0])),
        )
        print("\n🧮 Starting IK solver thread...")
        ik_thread = threading.Thread(target=ik_solver_thread, args=(data_manager, ik_solver), daemon=True)
        ik_thread.start()
        active_threads.append(ik_thread)

    if start_camera:
        print("\n📷 Starting camera thread...")
        cam_thread = threading.Thread(target=camera_thread, args=(data_manager,), daemon=True)
        cam_thread.start()
        active_threads.append(cam_thread)

    return data_manager, robot_controller, ik_solver, active_threads