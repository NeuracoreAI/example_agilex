#!/usr/bin/env python3
"""Minimal Piper Robot Policy Rollout (Headless).

This script serves as a lightweight, terminal-only entry point for executing
AI policies trained via Neuracore on the AgileX Piper robotic arm. It removes
the overhead of the Viser 3D GUI and Meta Quest tracking, making it ideal for
deployment on constrained compute environments or for automated testing.

Key Features:
    - Connects to Neuracore and loads policies (local, cloud run, or remote endpoint).
    - Bootstraps the core hardware control (CAN bus) and camera telemetry without GUI/IK overhead.
    - Extracts configuration settings dynamically from a YAML file.
    - Executes a continuous predict-and-act loop directly in the terminal.

Usage Examples:
    python 5_rollout_neuracore_policy_minimal.py --model-path ./model.nc.zip
    python 5_rollout_neuracore_policy_minimal.py --train-run-name my_awesome_training_run
    python 5_rollout_neuracore_policy_minimal.py --remote-endpoint-name live_cloud_endpoint
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import neuracore as nc

# ---------------------------------------------------------------------------
# Path Configuration & Local Imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config_parser import load_ik_config
from common.configs import (
    CAMERA_NAMES,
    GRIPPER_NAME,
    POLICY_EXECUTION_RATE,
    PREDICTION_HORIZON_EXECUTION_RATIO,
    URDF_PATH,
)
from common.data_manager import RobotActivityState
from common.policy_actions import run_policy
from common.policy_helpers import (
    embodiment_names_ordered,
    get_policy_embodiments,
    gripper_open_at_index,
    joint_targets_deg_at_index,
    log_robot_state_for_policy,
    print_policy_embodiments,
)
from common.policy_state import PolicyState
from common.system_bootstrap import bootstrap_robot_system
from neuracore_types import DataType


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def execute_horizon(
    data_manager: Any,
    policy_state: PolicyState,
    robot_controller: Any,
    frequency: int,
    input_embodiment: dict,
    output_grippers: list[str] | None,
) -> None:
    """Minimal terminal-only execution loop for the prediction horizon."""
    policy_state.start_policy_execution()
    data_manager.set_robot_activity_state(RobotActivityState.POLICY_CONTROLLED)

    locked_horizon = policy_state.get_locked_prediction_horizon()
    horizon_length = policy_state.get_locked_prediction_horizon_length()
    dt = 1.0 / frequency

    for i in range(horizon_length):
        start_time = time.time()

        # Dispatch joint targets
        joint_targets_deg = joint_targets_deg_at_index(locked_horizon, i)
        if joint_targets_deg is not None:
            robot_controller.set_target_joint_angles(joint_targets_deg)

        # Dispatch gripper targets
        gripper_target = gripper_open_at_index(
            locked_horizon, i, gripper_names=output_grippers
        )
        if gripper_target is not None:
            robot_controller.set_gripper_open_value(gripper_target)

        # Log state back to the policy for the next inference cycle
        log_robot_state_for_policy(data_manager, input_embodiment)

        # Maintain loop frequency
        time.sleep(max(0.0, dt - (time.time() - start_time)))

    policy_state.end_policy_execution()
    data_manager.set_robot_activity_state(RobotActivityState.ENABLED)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Piper Policy Test")

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

    parser.add_argument("--robot-name", type=str, default="AgileX PiPER")
    parser.add_argument("--frequency", type=int, default=POLICY_EXECUTION_RATE)
    parser.add_argument(
        "--execution-ratio", type=float, default=PREDICTION_HORIZON_EXECUTION_RATIO
    )
    parser.add_argument(
        "--ik-config",
        type=str,
        default="ik_conf/default.yaml",
        help="Path to YAML configuration.",
    )
    args = parser.parse_args()

    print("=" * 60 + "\nPIPER POLICY ROLLOUT (MINIMAL)\n" + "=" * 60)

    # 1. Configuration & Neuracore Connection
    config = load_ik_config(args.ik_config)
    nc.login()
    nc.connect_robot(
        robot_name=args.robot_name, urdf_path=str(URDF_PATH), overwrite=False
    )

    # 2. Policy Loading
    if args.remote_endpoint_name:
        print(
            f"\n🤖 Connecting to remote policy endpoint: {args.remote_endpoint_name}..."
        )
        try:
            policy = nc.policy_remote_server(args.remote_endpoint_name)
        except Exception:
            # Catch the EndpointError and provide explicit, actionable instructions
            print("\n" + "!" * 60)
            print(f"❌ ENDPOINT NOT ACTIVE: '{args.remote_endpoint_name}'")
            print("!" * 60)
            print(
                "The script successfully reached Neuracore, but the remote server is down."
            )
            print("\nPLEASE FOLLOW THESE STEPS:")
            print("  1. Open your browser and go to the Neuracore website/dashboard.")
            print(
                f"  2. Locate your deployment endpoint named '{args.remote_endpoint_name}'."
            )
            print("  3. Click 'Deploy' or 'Activate' to spin up the cloud server.")
            print(
                "  4. Wait for the status to show as 'Active', then rerun this script."
            )
            print("!" * 60 + "\n")
            sys.exit(1)

    elif args.train_run_name:
        print(f"\n🤖 Loading policy from cloud training run: {args.train_run_name}...")
        policy = nc.policy(
            train_run_name=args.train_run_name,
            device="cuda",
            robot_name=args.robot_name,
        )
    else:
        print(f"\n🤖 Loading policy from local model: {args.model_path}...")
        policy = nc.policy(
            model_file=args.model_path, device="cuda", robot_name=args.robot_name
        )

    # 3. Embodiment Extraction with Remote Fallback
    try:
        input_emb, output_emb = get_policy_embodiments(policy)
    except AttributeError:
        print(
            "\n⚠️  Could not dynamically extract embodiments. Using default Piper configuration..."
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

    output_gripper_names: list[str] | None = None
    if output_emb and DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS in output_emb:
        output_gripper_names = embodiment_names_ordered(
            output_emb[DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS]
        )

    # 4. Bootstrap Core System (No Quest, No IK needed for minimal playback)
    data_manager, robot_controller, ik_solver, active_threads = bootstrap_robot_system(
        config, start_ik=False, start_camera=True
    )

    policy_state = PolicyState()
    policy_state.set_execution_ratio(args.execution_ratio)

    time.sleep(2.0)  # Wait for hardware interfaces and threads to initialize

    # 5. Continuous Execution Loop
    try:
        print("\n🟢 Enabling and homing robot...")
        robot_controller.resume_robot()
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
        robot_controller.move_to_home()

        while (
            data_manager.get_robot_activity_state() == RobotActivityState.HOMING
            and not robot_controller.is_robot_homed()
        ):
            time.sleep(0.1)

        data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
        print(
            "✓ Robot ready. Starting policy execution loop... Press Ctrl+C to stop.\n"
        )

        while True:
            # Query the network for the next prediction
            if not run_policy(
                data_manager,
                policy,
                policy_state,
                visualizer=None,
                input_embodiment_description=input_emb,
            ):
                print("⚠️  Policy inference failed or timed out. Retrying...")
                time.sleep(0.5)
                continue

            # Execute the predicted horizon on the hardware
            execute_horizon(
                data_manager,
                policy_state,
                robot_controller,
                args.frequency,
                input_emb,
                output_gripper_names,
            )

    except KeyboardInterrupt:
        print("\n👋 Interrupt received. Halting execution loop...")
    except Exception as e:
        print(f"\n❌ Unhandled error during execution: {e}")
        traceback.print_exc()

    # 6. Graceful Teardown & Cleanup
    finally:
        print("\n🧹 Cleaning up subsystems...")
        try:
            data_manager.set_robot_activity_state(RobotActivityState.HOMING)
            robot_controller.move_to_home()
            # Wait safely for homing to finish before severing the hardware connection
            for _ in range(10):
                if robot_controller.is_robot_homed():
                    break
                time.sleep(1)
        except Exception:
            pass

        policy.disconnect()
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        data_manager.request_shutdown()

        for t in active_threads:
            t.join(timeout=2.0)

        if robot_controller:
            try:
                robot_controller.graceful_stop()
            except AttributeError:
                pass
            robot_controller.cleanup()

        nc.logout()
        print("👋 Shutdown complete.")
