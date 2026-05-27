#!/usr/bin/env python3
"""
Piper Robot Policy Rollout and Visualization.

This script serves as a production-ready entry point for executing and visualizing
AI policies trained via Neuracore on the AgileX Piper robotic arm. It handles:
    1. Establishing a connection to the Neuracore API and authenticating the hardware.
    2. Dynamically loading a trained policy via a local model, a training run, or a remote endpoint.
    3. Extracting and validating the expected input/output embodiment descriptions.
    4. Bootstrapping the core robot control and background state-management threads using a YAML config.
    5. Launching a Viser-based web UI for real-time 3D visualization and manual policy control.
    6. Managing background threads dedicated to continuous policy prediction and execution.

Usage Examples:
    python 4_rollout_neuracore_policy.py --model-path ./model.nc.zip --ik-config ik_conf/default.yaml
    python 4_rollout_neuracore_policy.py --remote-endpoint-name test_deploy
"""

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path

import neuracore as nc

# Dynamically append the parent directory to sys.path to resolve local 'common' modules.
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config_parser import load_ik_config
from common.configs import (
    CAMERA_NAMES,
    GRIPPER_NAME,
    POLICY_EXECUTION_RATE,
    PREDICTION_HORIZON_EXECUTION_RATIO,
    ROBOT_RATE,
    URDF_PATH,
)
from common.data_manager import RobotActivityState

# Extracted policy execution and lifecycle management actions
from common.policy_actions import (
    play_policy,
    policy_execution_thread,
    run_policy,
    start_policy_execution,
)
from common.policy_helpers import (
    get_policy_embodiments,
    print_policy_embodiments,
)
from common.policy_state import PolicyState
from common.robot_visualizer import RobotVisualizer
from common.shared_actions import move_robot_home, toggle_robot_enabled
from common.system_bootstrap import bootstrap_robot_system
from neuracore_types import DataType

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. Argument Parsing
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Execute and visualize Neuracore policies on the Piper Robot."
    )

    parser.add_argument(
        "--continuous-mode",
        choices=["pipeline", "sequential"],
        default="sequential",
        help="Execution strategy for the receding horizon loop.",
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default="AgileX PiPER",
        help="The registered hardware name in the Neuracore ecosystem.",
    )
    parser.add_argument(
        "--ik-config",
        type=str,
        default="ik_conf/default.yaml",
        help="Path to IK/teleop YAML configuration file.",
    )

    # Require exactly one method of loading the policy
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument(
        "--train-run-name", type=str, default=None, help="Cloud training run name."
    )
    policy_group.add_argument(
        "--model-path", type=str, default=None, help="Path to local .nc.zip model file."
    )
    policy_group.add_argument(
        "--remote-endpoint-name",
        type=str,
        default=None,
        help="Active remote inference endpoint.",
    )

    args = parser.parse_args()

    print("=" * 60 + "\nPIPER ROBOT TEST WITH NEURACORE POLICY\n" + "=" * 60)

    # ---------------------------------------------------------
    # 2. Configuration & Neuracore Initialization
    # ---------------------------------------------------------
    # Load the YAML configuration dictionary
    config = load_ik_config(args.ik_config)

    nc.login()
    nc.connect_robot(
        robot_name=args.robot_name, urdf_path=str(URDF_PATH), overwrite=False
    )

    # ---------------------------------------------------------
    # 3. Policy Loading & Embodiment Resolution
    # ---------------------------------------------------------
    if args.remote_endpoint_name is not None:
        print(
            f"\n🤖 Connecting to remote policy endpoint: {args.remote_endpoint_name}..."
        )
        try:
            policy = nc.policy_remote_server(args.remote_endpoint_name)
        except nc.EndpointError:
            print(
                f"❌ Endpoint '{args.remote_endpoint_name}' not available. "
                "Please start it from the Neuracore dashboard."
            )
            sys.exit(1)

    elif args.train_run_name is not None:
        print(f"\n🤖 Loading policy from cloud training run: {args.train_run_name}...")
        policy = nc.policy(
            train_run_name=args.train_run_name,
            device="cuda",
            robot_name=args.robot_name,
        )

    else:
        print(f"\n🤖 Loading policy from local model: {args.model_path}...")
        policy = nc.policy(
            model_file=args.model_path,
            device="cuda",
            robot_name=args.robot_name,
        )

    # Dynamically extract what sensor streams the model expects to see.
    # Remote endpoints (policy_remote_server) do not expose these attributes,
    # so we catch the AttributeError and fall back to the default Piper embodiment.
    try:
        input_emb, output_emb = get_policy_embodiments(policy)
    except AttributeError:
        print(
            "\n⚠️  Could not dynamically extract embodiments from remote endpoint. Using default Piper configuration..."
        )
        input_emb = {
            DataType.JOINT_POSITIONS: {i: f"joint{i+1}" for i in range(6)},
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: {0: GRIPPER_NAME},
            DataType.RGB_IMAGES: {0: CAMERA_NAMES[0]},
        }
        output_emb = {
            DataType.JOINT_TARGET_POSITIONS: {i: f"joint{i+1}" for i in range(6)},
            DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS: {0: GRIPPER_NAME},
        }

    print_policy_embodiments(input_emb, output_emb)

    # Initialize shared policy state container
    policy_state = PolicyState()
    policy_state.set_execution_mode(PolicyState.ExecutionMode.TARGETING_TIME)

    # ---------------------------------------------------------
    # 4. Hardware & Subsystem Bootstrapping
    # ---------------------------------------------------------
    # Instantiates the shared DataManager, CAN hardware interface, IK solver,
    # and base telemetry threads using the parsed YAML config parameters.
    data_manager, robot_controller, ik_solver, active_threads = bootstrap_robot_system(
        config, start_ik=True, start_camera=True
    )

    # ---------------------------------------------------------
    # 5. Visualization & UI Setup
    # ---------------------------------------------------------
    print("\n🖥️  Starting Viser visualization server...")
    visualizer = RobotVisualizer(str(URDF_PATH))

    # Inject UI components using fixed AI policy rates
    visualizer.add_policy_controls(
        PREDICTION_HORIZON_EXECUTION_RATIO,
        POLICY_EXECUTION_RATE,
        ROBOT_RATE,
        "targeting_time",
    )
    visualizer.add_toggle_robot_enabled_status_button()
    visualizer.add_homing_controls()
    visualizer.add_policy_buttons()

    # Bind hardware action callbacks to the UI buttons
    visualizer.set_toggle_robot_enabled_status_callback(
        lambda: toggle_robot_enabled(data_manager, robot_controller, visualizer)
    )
    visualizer.set_go_home_callback(
        lambda: move_robot_home(data_manager, robot_controller)
    )

    # Bind policy execution callbacks to the UI buttons
    visualizer.set_run_policy_callback(
        lambda: run_policy(data_manager, policy, policy_state, visualizer, input_emb)
    )
    visualizer.set_start_policy_execution_callback(
        lambda: start_policy_execution(data_manager, policy_state)
    )
    visualizer.set_play_policy_callback(
        lambda: play_policy(
            data_manager,
            policy,
            policy_state,
            visualizer,
            input_emb,
            args.continuous_mode,
        )
    )
    visualizer.set_execution_mode_callback(
        lambda: policy_state.set_execution_mode(
            PolicyState.ExecutionMode(visualizer.get_execution_mode())
        )
    )

    # ---------------------------------------------------------
    # 6. Dispatch Policy Execution Thread
    # ---------------------------------------------------------
    policy_exec_thread = threading.Thread(
        target=policy_execution_thread,
        args=(
            policy,
            data_manager,
            policy_state,
            robot_controller,
            visualizer,
            input_emb,
        ),
        daemon=True,
        name="PolicyExecutionWorker",
    )
    policy_exec_thread.start()
    active_threads.append(policy_exec_thread)

    print(
        "\n🚀 System Online! Open http://localhost:8080 in your browser to visualize and run the policy.\n"
    )

    # ---------------------------------------------------------
    # 7. Main Daemon Loop
    # ---------------------------------------------------------
    try:
        # Keep the main thread alive. All heavy lifting is handled via background threads.
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n👋 Interrupt signal received. Initiating graceful shutdown...")
    except Exception as e:
        print(f"\n❌ Unhandled Exception in main loop: {e}")
        traceback.print_exc()

    # ---------------------------------------------------------
    # 8. Graceful Teardown & Cleanup
    # ---------------------------------------------------------
    print("\n🧹 Cleaning up subsystems...")

    # Safely sever the connection to the policy backend
    policy.disconnect()

    # Broadcast shutdown signal to all worker threads
    data_manager.request_shutdown()
    data_manager.set_robot_activity_state(RobotActivityState.DISABLED)

    # Wait for daemon threads to finish their current execution cycles
    for thread in active_threads:
        thread.join(timeout=2.0)

    # Relinquish hardware interfaces and network ports
    robot_controller.cleanup()
    visualizer.stop()
    nc.logout()

    print("👋 Shutdown complete. Goodbye.")
