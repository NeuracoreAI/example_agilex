#!/usr/bin/env python3
"""Minimal Piper Robot Policy Test - Terminal only, no GUI."""

import argparse
import sys
import time
import traceback
from pathlib import Path
import neuracore as nc

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import URDF_PATH, POLICY_EXECUTION_RATE, PREDICTION_HORIZON_EXECUTION_RATIO
from common.data_manager import RobotActivityState
from common.policy_helpers import get_policy_embodiments, print_policy_embodiments, embodiment_names_ordered, gripper_open_at_index, joint_targets_deg_at_index, log_robot_state_for_policy
from common.policy_state import PolicyState
from common.system_bootstrap import bootstrap_robot_system
from common.policy_actions import run_policy
from neuracore_types import DataType

def execute_horizon(data_manager, policy_state, robot_controller, frequency, input_embodiment, output_grippers):
    """Minimal terminal-only execution loop for the prediction horizon."""
    policy_state.start_policy_execution()
    data_manager.set_robot_activity_state(RobotActivityState.POLICY_CONTROLLED)

    locked_horizon = policy_state.get_locked_prediction_horizon()
    horizon_length = policy_state.get_locked_prediction_horizon_length()
    dt = 1.0 / frequency

    for i in range(horizon_length):
        start_time = time.time()

        joint_targets_deg = joint_targets_deg_at_index(locked_horizon, i)
        if joint_targets_deg is not None:
            robot_controller.set_target_joint_angles(joint_targets_deg)

        gripper_target = gripper_open_at_index(locked_horizon, i, gripper_names=output_grippers)
        if gripper_target is not None:
            robot_controller.set_gripper_open_value(gripper_target)

        log_robot_state_for_policy(data_manager, input_embodiment)
        time.sleep(max(0, dt - (time.time() - start_time)))

    policy_state.end_policy_execution()
    data_manager.set_robot_activity_state(RobotActivityState.ENABLED)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Piper Policy Test")
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument("--train-run-name", type=str, default=None)
    policy_group.add_argument("--model-path", type=str, default=None)
    policy_group.add_argument("--remote-endpoint-name", type=str, default=None)
    parser.add_argument("--robot-name", type=str, default="AgileX PiPER")
    parser.add_argument("--frequency", type=int, default=POLICY_EXECUTION_RATE)
    parser.add_argument("--execution-ratio", type=float, default=PREDICTION_HORIZON_EXECUTION_RATIO)
    args = parser.parse_args()

    print("=" * 60 + "\nPIPER POLICY ROLLOUT (MINIMAL)\n" + "=" * 60)

    # 1. Connect & Load Policy
    nc.login()
    nc.connect_robot(robot_name=args.robot_name, urdf_path=str(URDF_PATH), overwrite=False)

    if args.remote_endpoint_name:
        policy = nc.policy_remote_server(args.remote_endpoint_name)
    elif args.train_run_name:
        policy = nc.policy(train_run_name=args.train_run_name, device="cuda", robot_name=args.robot_name)
    else:
        policy = nc.policy(model_file=args.model_path, device="cuda", robot_name=args.robot_name)
    
    input_emb, output_emb = get_policy_embodiments(policy)
    print_policy_embodiments(input_emb, output_emb)
    
    output_gripper_names = None
    if output_emb and DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS in output_emb:
        output_gripper_names = embodiment_names_ordered(output_emb[DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS])

    # 2. Bootstrap Core System (No Quest, No IK needed for minimal playback)
    data_manager, robot_controller, _, active_threads = bootstrap_robot_system(start_ik=False, start_camera=True)
    policy_state = PolicyState()
    policy_state.set_execution_ratio(args.execution_ratio)

    time.sleep(2.0) # Wait for threads to init

    try:
        print("\n🟢 Enabling and homing robot...")
        robot_controller.resume_robot()
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
        robot_controller.move_to_home()
        
        while data_manager.get_robot_activity_state() == RobotActivityState.HOMING and not robot_controller.is_robot_homed():
            time.sleep(0.1)
            
        data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
        print("✓ Robot ready. Starting policy execution loop... Press Ctrl+C to stop.\n")

        # 3. Continuous Execution Loop
        while True:
            if not run_policy(data_manager, policy, policy_state, visualizer=None, input_embodiment_description=input_emb):
                print("⚠️  Policy run failed, retrying...")
                time.sleep(0.5)
                continue
            
            execute_horizon(data_manager, policy_state, robot_controller, args.frequency, input_emb, output_gripper_names)

    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
    finally:
        print("\n🧹 Cleaning up...")
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
        robot_controller.move_to_home()
        time.sleep(1.0)
        
        policy.disconnect()
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        data_manager.request_shutdown()
        for t in active_threads: 
            t.join()
        robot_controller.cleanup()
        nc.logout()