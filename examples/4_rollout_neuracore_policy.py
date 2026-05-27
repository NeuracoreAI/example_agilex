#!/usr/bin/env python3
"""
Piper Robot Policy Rollout and Visualization.

This script serves as a production-ready entry point for executing and visualizing
AI policies trained via Neuracore on the AgileX Piper robotic arm. It handles:
    1. Establishing a connection to the Neuracore API and authenticating the hardware.
    2. Dynamically loading a trained policy via a local model, a training run, or a remote endpoint.
    3. Extracting and validating the expected input/output embodiment descriptions.
    4. Bootstrapping the core robot control and background state-management threads.
    5. Launching a Viser-based web UI for real-time 3D visualization and manual policy control.
    6. Managing background threads dedicated to continuous policy prediction and execution.

Usage Examples:
    python 4_rollout_neuracore_policy.py --model-path ./model.nc.zip
    python 4_rollout_neuracore_policy.py --train-run-name my_awesome_training_run
    python 4_rollout_neuracore_policy.py --remote-endpoint-name live_cloud_endpoint
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

from common.configs import (
    CAMERA_NAMES, GRIPPER_NAME, URDF_PATH,
    PREDICTION_HORIZON_EXECUTION_RATIO, POLICY_EXECUTION_RATE, ROBOT_RATE,
)
from neuracore_types import DataType, EmbodimentDescription

from common.data_manager import RobotActivityState
from common.policy_state import PolicyState
from common.policy_helpers import (
    embodiment_names_ordered, 
    get_policy_embodiments, 
    print_policy_embodiments
)
from common.system_bootstrap import bootstrap_robot_system
from common.shared_actions import toggle_robot_enabled, move_robot_home
from common.robot_visualizer import RobotVisualizer

# Extracted policy execution and lifecycle management actions
from common.policy_actions import (
    run_policy, start_policy_execution, play_policy, policy_execution_thread
)

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
        help="Execution strategy for the receding horizon loop."
    )
    parser.add_argument(
        "--robot-name", 
        type=str, 
        default="AgileX PiPER",
        help="The registered hardware name in the Neuracore ecosystem."
    )
    
    # Require exactly one method of loading the policy
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument("--train-run-name", type=str, default=None, help="Cloud training run name.")
    policy_group.add_argument("--model-path", type=str, default=None, help="Path to local .nc.zip model file.")
    policy_group.add_argument("--remote-endpoint-name", type=str, default=None, help="Active remote inference endpoint.")
    
    args = parser.parse_args()

    print("=" * 60 + "\nPIPER ROBOT TEST WITH NEURACORE POLICY\n" + "=" * 60)

    # ---------------------------------------------------------
    # 2. Neuracore Initialization & Authentication
    # ---------------------------------------------------------
    nc.login()
    nc.connect_robot(robot_name=args.robot_name, urdf_path=str(URDF_PATH), overwrite=False)

    # ---------------------------------------------------------
    # 3. Policy Loading & Embodiment Resolution
    # ---------------------------------------------------------
    if args.remote_endpoint_name is not None:
        print(f"\n🤖 Connecting to remote policy endpoint: {args.remote_endpoint_name}...")
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

    # Dynamically extract and log what sensor streams the model expects to see
    input_emb, output_emb = get_policy_embodiments(policy)
    print_policy_embodiments(input_emb, output_emb)

    # Initialize shared policy state container
    policy_state = PolicyState()
    policy_state.set_execution_mode(PolicyState.ExecutionMode.TARGETING_TIME)

    # ---------------------------------------------------------
    # 4. Hardware & Subsystem Bootstrapping
    # ---------------------------------------------------------
    # Instantiates the shared DataManager, CAN hardware interface, IK solver, 
    # and base telemetry threads (joint states, camera streams).
    data_manager, robot_controller, ik_solver, active_threads = bootstrap_robot_system(
        start_ik=True, start_camera=True
    )

    # ---------------------------------------------------------
    # 5. Visualization & UI Setup
    # ---------------------------------------------------------
    print("\n🖥️  Starting Viser visualization server...")
    visualizer = RobotVisualizer(str(URDF_PATH))
    
    # Inject UI components
    visualizer.add_policy_controls(
        PREDICTION_HORIZON_EXECUTION_RATIO, 
        POLICY_EXECUTION_RATE, 
        ROBOT_RATE, 
        "targeting_time"
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
        lambda: play_policy(data_manager, policy, policy_state, visualizer, input_emb, args.continuous_mode)
    )
    visualizer.set_execution_mode_callback(
        lambda: policy_state.set_execution_mode(PolicyState.ExecutionMode(visualizer.get_execution_mode()))
    )

    # ---------------------------------------------------------
    # 6. Dispatch Policy Execution Thread
    # ---------------------------------------------------------
    policy_exec_thread = threading.Thread(
        target=policy_execution_thread,
        args=(policy, data_manager, policy_state, robot_controller, visualizer, input_emb),
        daemon=True,
        name="PolicyExecutionWorker"
    )
    policy_exec_thread.start()
    active_threads.append(policy_exec_thread)

    print("\n🚀 System Online! Open http://localhost:8080 in your browser to visualize and run the policy.\n")

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