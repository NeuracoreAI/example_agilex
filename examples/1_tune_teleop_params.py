#!/usr/bin/env python3
"""
Piper Robot Teleoperation Tuning - Real Robot Control.

This script allows developers to iteratively tune the robot's teleoperation parameters
in real-time. It provides a Viser-based web UI where you can adjust Inverse Kinematics (IK)
weights, 1-Euro Filter lag/smoothness coefficients, and spatial scaling on the fly while
piloting the AgileX Piper arm using a Meta Quest controller.
"""

import argparse
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

# Add parent directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config_parser import load_ik_config

# Only import the fixed hardware constants that are left in configs.py
from common.configs import (
    CAMERA_NAMES,
    META_QUEST_AXIS_MASK,
    URDF_PATH,
    VISUALIZATION_RATE,
)
from common.data_manager import RobotActivityState
from common.robot_visualizer import RobotVisualizer
from common.shared_actions import move_robot_home, toggle_robot_enabled
from common.system_bootstrap import bootstrap_robot_system
from common.threads.quest_reader import quest_reader_thread
from meta_quest_teleop.reader import MetaQuestReader

from piper_controller import PiperController

# ---------------------------------------------------------------------------
# Dynamic Action Callbacks
# ---------------------------------------------------------------------------


def _step_wrist_joint(
    data_manager, robot_controller, visualizer, direction: int
) -> None:
    """Apply a relative manual step to the wrist joint using the dynamically tuned step degree."""
    data_manager.set_teleop_state(False, None, None)
    robot_controller.set_control_mode(PiperController.ControlMode.JOINT_SPACE)

    target_joint_angles = robot_controller.get_target_joint_angles()
    current_joint_angles = data_manager.get_current_joint_angles()

    base_wrist = (
        float(current_joint_angles[-1])
        if current_joint_angles is not None
        else float(target_joint_angles[-1])
    )

    # Read the tuned value directly from the UI and apply direction multiplier (1 or -1)
    delta_degrees = visualizer.get_wrist_step_degrees() * direction
    target_joint_angles[-1] = base_wrist + delta_degrees

    robot_controller.set_target_joint_angles(target_joint_angles)
    data_manager.set_target_joint_angles(target_joint_angles)


def toggle_slow_scaling(data_manager, visualizer):
    """Toggle the movement scaling mode between standard and dynamically-tuned precision."""
    enabled = data_manager.toggle_slow_scaling_mode_enabled()
    if enabled:
        print(
            f"🐢 Slow scaling enabled (trans={visualizer.get_slow_translation_scale():.3f}, rot={visualizer.get_slow_rotation_scale():.3f})"
        )
    else:
        print(
            f"🐇 Slow scaling disabled (trans={visualizer.get_translation_scale():.3f}, rot={visualizer.get_rotation_scale():.3f})"
        )


def save_config_to_yaml(
    visualizer, filepath="ik_conf/tuned_teleop_configs.yaml"
) -> None:
    """Extracts all current UI slider values and saves them to a YAML file."""
    try:
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        pink_params = visualizer.get_pink_parameters()
        min_c, beta, d_c = visualizer.get_controller_filter_params()

        # Cast to standard python floats/lists so PyYAML doesn't choke on numpy types
        config_dict = {
            "ik_parameters": {
                "position_cost": float(pink_params["position_cost"]),
                "orientation_cost": float(pink_params["orientation_cost"]),
                "frame_task_gain": float(pink_params["frame_task_gain"]),
                "lm_damping": float(pink_params["lm_damping"]),
                "damping_cost": float(pink_params["damping_cost"]),
                "solver_damping_value": float(pink_params["solver_damping_value"]),
                "posture_cost_vector": pink_params["posture_cost_vector"].tolist(),
            },
            "filter_parameters": {
                "controller_min_cutoff": float(min_c),
                "controller_beta": float(beta),
                "controller_d_cutoff": float(d_c),
            },
            "teleop_parameters": {
                "translation_scale": float(visualizer.get_translation_scale()),
                "rotation_scale": float(visualizer.get_rotation_scale()),
                "slow_translation_scale": float(
                    visualizer.get_slow_translation_scale()
                ),
                "slow_rotation_scale": float(visualizer.get_slow_rotation_scale()),
                "wrist_joint_button_step_degrees": float(
                    visualizer.get_wrist_step_degrees()
                ),
            },
        }

        with open(filepath, "w") as file:
            yaml.dump(config_dict, file, default_flow_style=False, sort_keys=False)
        print(
            f"\n✅ SUCCESS: Configuration securely saved to {os.path.abspath(filepath)}"
        )
    except Exception as e:
        print(f"\n❌ ERROR: Failed to save YAML config: {e}")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Piper Robot Teleoperation Tuning")
    parser.add_argument("--ip-address", type=str, default=None)
    parser.add_argument(
        "--ik-config",
        type=str,
        default="ik_conf/default.yaml",
        help="Path to IK/teleop YAML config.",
    )
    args = parser.parse_args()

    print("=" * 60 + "\nPIPER ROBOT TELEOPERATION TUNING\n" + "=" * 60)

    # Load the YAML dict
    config = load_ik_config(args.ik_config)
    ik_p = config.get("ik_parameters", {})
    filt_p = config.get("filter_parameters", {})
    tele_p = config.get("teleop_parameters", {})

    # 1. Bootstrap Core System
    data_manager, robot_controller, ik_solver, active_threads = bootstrap_robot_system(
        config, start_ik=True, start_camera=True
    )

    # 2. Initialize Quest Reader & Handle ADB Exceptions
    print("\n🎮 Initializing Meta Quest reader...")
    try:
        quest_reader = MetaQuestReader(
            ip_address=args.ip_address,
            port=5555,
            run=True,
            axis_mask=META_QUEST_AXIS_MASK,
        )
    except (Exception, SystemExit):
        print("\n" + "!" * 60)
        print("❌ FAILED TO ACCESS META QUEST")
        print("!" * 60)
        print("The headset is plugged in, but ADB debugging permissions are missing.")
        print("\nPLEASE FOLLOW THESE STEPS:")
        print("  1. Put the Meta Quest headset on your head.")
        print("  2. Look for a notification in your menu that says 'USB Detected'.")
        print(
            "  3. Click on that notification and select 'Allow' to grant data access."
        )
        print("  4. Rerun this script.")
        print("!" * 60 + "\n")

        data_manager.request_shutdown()
        for t in active_threads:
            t.join(timeout=1.0)
        robot_controller.cleanup()
        sys.exit(1)

    # 3. Initialize Visualizer Web UI
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

    # Add tuning sliders
    visualizer.add_controller_filter_controls(
        filt_p.get("controller_min_cutoff", 0.8),
        filt_p.get("controller_beta", 0.05),
        filt_p.get("controller_d_cutoff", 0.9),
    )
    visualizer.add_scaling_controls(
        tele_p.get("translation_scale", 1.5), tele_p.get("rotation_scale", 1.2)
    )
    visualizer.add_pink_parameter_controls(
        ik_p.get("position_cost", 1.0),
        ik_p.get("orientation_cost", 0.75),
        ik_p.get("frame_task_gain", 0.4),
        ik_p.get("lm_damping", 0.01),
        ik_p.get("damping_cost", 0.25),
        ik_p.get("solver_damping_value", 1e-4),
        ik_p.get("posture_cost_vector", [0.0, 0.0, 0.0, 0.05, 0.0, 0.0]),
    )
    visualizer.add_advanced_tuning_controls(
        tele_p.get("slow_translation_scale", 0.6),
        tele_p.get("slow_rotation_scale", 0.6),
        tele_p.get("wrist_joint_button_step_degrees", 5.0),
    )

    # 4. Bind Shared Callbacks (Quest Buttons & GUI Buttons)
    quest_reader.on(
        "button_a_pressed",
        lambda: toggle_robot_enabled(data_manager, robot_controller, visualizer),
    )
    quest_reader.on(
        "button_b_pressed", lambda: move_robot_home(data_manager, robot_controller)
    )
    quest_reader.on(
        "button_y_pressed",
        lambda: _step_wrist_joint(
            data_manager, robot_controller, visualizer, direction=1
        ),
    )
    quest_reader.on(
        "button_x_pressed",
        lambda: _step_wrist_joint(
            data_manager, robot_controller, visualizer, direction=-1
        ),
    )
    quest_reader.on(
        "button_lj_pressed", lambda: toggle_slow_scaling(data_manager, visualizer)
    )

    visualizer.set_toggle_robot_enabled_status_callback(
        lambda: toggle_robot_enabled(data_manager, robot_controller, visualizer)
    )
    visualizer.set_go_home_callback(
        lambda: move_robot_home(data_manager, robot_controller)
    )

    # Safely bind the save config callback with the correct arguments
    visualizer.set_save_config_callback(
        lambda: save_config_to_yaml(
            visualizer, filepath="ik_conf/tuned_teleop_configs.yaml"
        )
    )

    # 5. Start Quest Thread
    quest_thread = threading.Thread(
        target=quest_reader_thread, args=(data_manager, quest_reader), daemon=True
    )
    quest_thread.start()
    active_threads.append(quest_thread)

    print(
        "\n🚀 Ready! Use Meta Quest controllers to move the robot. Open http://localhost:8080 to tune. Press Ctrl+C to exit.\n"
    )

    # 6. Main Visualization Loop
    dt = 1.0 / VISUALIZATION_RATE
    try:
        while True:
            iteration_start = time.time()

            # Poll GUI for real-time scale tuning updates
            t_scale = (
                visualizer.get_slow_translation_scale()
                if data_manager.get_slow_scaling_mode_enabled()
                else visualizer.get_translation_scale()
            )
            r_scale = (
                visualizer.get_slow_rotation_scale()
                if data_manager.get_slow_scaling_mode_enabled()
                else visualizer.get_rotation_scale()
            )
            data_manager.set_teleop_scaling(t_scale, r_scale)

            # Retrieve State Data
            controller_transform, grip_value, trigger_value = (
                data_manager.get_controller_data()
            )
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
                (
                    controller_transform[:3, 3]
                    if controller_transform is not None
                    else None
                ),
                connected=controller_transform is not None,
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
            visualizer.update_gripper_status(
                trigger_value, robot_enabled=(state == RobotActivityState.ENABLED)
            )

            # Maintain UI refresh rate
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
        t.join(timeout=2.0)
    robot_controller.cleanup()
    visualizer.stop()
    print("👋 Demo stopped.")
