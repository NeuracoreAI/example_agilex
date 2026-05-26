#!/usr/bin/env python3
"""Piper Robot Teleoperation with Meta Quest Controller - REAL ROBOT CONTROL."""

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path
import numpy as np

# Add parent directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import (
    CAMERA_NAMES, META_QUEST_AXIS_MASK, ROTATION_SCALE, 
    SLOW_ROTATION_SCALE, SLOW_TRANSLATION_SCALE, TRANSLATION_SCALE, 
    URDF_PATH, VISUALIZATION_RATE, WRIST_JOINT_BUTTON_STEP_DEGREES,
    CONTROLLER_MIN_CUTOFF, CONTROLLER_BETA, CONTROLLER_D_CUTOFF,
    POSITION_COST, ORIENTATION_COST, FRAME_TASK_GAIN, LM_DAMPING, 
    DAMPING_COST, SOLVER_DAMPING_VALUE, POSTURE_COST_VECTOR
)
from common.data_manager import RobotActivityState
from common.system_bootstrap import bootstrap_robot_system
from common.shared_actions import toggle_robot_enabled, move_robot_home
from common.robot_visualizer import RobotVisualizer
from common.threads.quest_reader import quest_reader_thread
from meta_quest_teleop.reader import MetaQuestReader
from piper_controller import PiperController


def _step_wrist_joint(data_manager, robot_controller, delta_degrees: float) -> None:
    """Apply a relative step to the wrist joint target angle."""
    data_manager.set_teleop_state(False, None, None)
    robot_controller.set_control_mode(PiperController.ControlMode.JOINT_SPACE)

    target_joint_angles = robot_controller.get_target_joint_angles()
    current_joint_angles = data_manager.get_current_joint_angles()
    base_wrist = float(current_joint_angles[-1]) if current_joint_angles is not None else float(target_joint_angles[-1])

    target_joint_angles[-1] = base_wrist + delta_degrees
    robot_controller.set_target_joint_angles(target_joint_angles)
    data_manager.set_target_joint_angles(target_joint_angles)


def toggle_slow_scaling(data_manager):
    """Toggle the movement scaling mode."""
    enabled = data_manager.toggle_slow_scaling_mode_enabled()
    if enabled:
        print(f"🐢 Slow scaling enabled (trans={SLOW_TRANSLATION_SCALE:.3f}, rot={SLOW_ROTATION_SCALE:.3f})")
    else:
        print(f"🐇 Slow scaling disabled (trans={TRANSLATION_SCALE:.3f}, rot={ROTATION_SCALE:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Piper Robot Teleoperation")
    parser.add_argument("--ip-address", type=str, default=None, help="Meta Quest IP")
    args = parser.parse_args()

    print("=" * 60 + "\nPIPER ROBOT TELEOPERATION\n" + "=" * 60)

    # 1. Bootstrap Core System (Data Manager, Robot Controller, IK Solver, Base Threads)
    data_manager, robot_controller, ik_solver, active_threads = bootstrap_robot_system(
        start_ik=True, start_camera=True
    )

    # 2. Initialize Quest Reader
    print("\n🎮 Initializing Meta Quest reader...")
    quest_reader = MetaQuestReader(
        ip_address=args.ip_address, port=5555, run=True, axis_mask=META_QUEST_AXIS_MASK
    )
    
    # 3. Initialize Visualizer
    print("\n🖥️  Starting visualization...")
    visualizer = RobotVisualizer(urdf_path=URDF_PATH)
    visualizer.add_basic_controls()
    visualizer.add_teleop_controls()
    visualizer.add_gripper_status_controls()
    visualizer.add_homing_controls()
    visualizer.add_toggle_robot_enabled_status_button()
    visualizer.add_rgb_image_placeholder()
    visualizer.add_target_frame_visualization()
    visualizer.add_controller_visualization()
    visualizer.add_controller_filter_controls(CONTROLLER_MIN_CUTOFF, CONTROLLER_BETA, CONTROLLER_D_CUTOFF)
    visualizer.add_scaling_controls(TRANSLATION_SCALE, ROTATION_SCALE)
    visualizer.add_pink_parameter_controls(
        POSITION_COST, ORIENTATION_COST, FRAME_TASK_GAIN, LM_DAMPING, 
        DAMPING_COST, SOLVER_DAMPING_VALUE, POSTURE_COST_VECTOR
    )

    # 4. Bind Shared Callbacks (Quest Buttons & GUI Buttons)
    quest_reader.on("button_a_pressed", lambda: toggle_robot_enabled(data_manager, robot_controller, visualizer))
    quest_reader.on("button_b_pressed", lambda: move_robot_home(data_manager, robot_controller))
    quest_reader.on("button_y_pressed", lambda: _step_wrist_joint(data_manager, robot_controller, WRIST_JOINT_BUTTON_STEP_DEGREES))
    quest_reader.on("button_x_pressed", lambda: _step_wrist_joint(data_manager, robot_controller, -WRIST_JOINT_BUTTON_STEP_DEGREES))
    quest_reader.on("button_lj_pressed", lambda: toggle_slow_scaling(data_manager))
    
    visualizer.set_toggle_robot_enabled_status_callback(lambda: toggle_robot_enabled(data_manager, robot_controller, visualizer))
    visualizer.set_go_home_callback(lambda: move_robot_home(data_manager, robot_controller))

    # 5. Start Quest Thread
    quest_thread = threading.Thread(target=quest_reader_thread, args=(data_manager, quest_reader), daemon=True)
    quest_thread.start()
    active_threads.append(quest_thread)

    print("\n🚀 Ready! Use Meta Quest controllers to move the robot. Press Ctrl+C to exit.\n")

    # 6. Main Visualization Loop
    dt = 1.0 / VISUALIZATION_RATE
    try:
        while True:
            iteration_start = time.time()

            # Pass GUI scaling updates to Data Manager
            t_scale = SLOW_TRANSLATION_SCALE if data_manager.get_slow_scaling_mode_enabled() else visualizer.get_translation_scale()
            r_scale = SLOW_ROTATION_SCALE if data_manager.get_slow_scaling_mode_enabled() else visualizer.get_rotation_scale()
            data_manager.set_teleop_scaling(t_scale, r_scale)

            # Retrieve State Data
            controller_transform, grip_value, trigger_value = data_manager.get_controller_data()
            teleop_active = data_manager.get_teleop_active()
            state = data_manager.get_robot_activity_state()
            current_joints = data_manager.get_current_joint_angles()
            target_joints = data_manager.get_target_joint_angles()
            rgb_image = data_manager.get_rgb_image(CAMERA_NAMES[0])

            # Update GUI Displays
            visualizer.set_grip_value(grip_value)
            visualizer.set_trigger_value(trigger_value)
            visualizer.update_timing(data_manager.get_ik_solve_time_ms())
            visualizer.update_controller_visualization(controller_transform)
            visualizer.update_controller_status_display(
                controller_transform[:3, 3] if controller_transform is not None else None, 
                connected=controller_transform is not None
            )
            visualizer.update_teleop_status(teleop_active)
            visualizer.update_target_visualization(data_manager.get_target_pose())
            visualizer.update_rgb_image(rgb_image)
            
            # Render Actual Robot
            if current_joints is not None:
                rad_joints = np.radians(current_joints)
                visualizer.update_robot_pose(rad_joints)
                visualizer.update_joint_angles_display(rad_joints)

            # Render Ghost Robot Target
            if target_joints is not None and state == RobotActivityState.ENABLED:
                visualizer.update_ghost_robot_visibility(True)
                visualizer.update_ghost_robot_pose(np.radians(target_joints))
            else:
                visualizer.update_ghost_robot_visibility(False)

            visualizer.update_robot_status(f"Robot Status: {state.value.capitalize()}")
            visualizer.update_gripper_status(trigger_value, robot_enabled=(state == RobotActivityState.ENABLED))

            # Maintain Loop Rate
            time.sleep(max(0, dt - (time.time() - iteration_start)))

    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()

    # 7. Cleanup
    print("\n🧹 Cleaning up...")
    data_manager.request_shutdown()
    data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
    quest_reader.stop()
    for t in active_threads: 
        t.join()
    robot_controller.cleanup()
    visualizer.stop()
    print("👋 Demo stopped.")