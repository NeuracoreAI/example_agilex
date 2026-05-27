#!/usr/bin/env python3
"""Piper Robot Test with Neuracore policy."""

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path
import neuracore as nc

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import (
    CAMERA_NAMES, GRIPPER_NAME, URDF_PATH,
    PREDICTION_HORIZON_EXECUTION_RATIO, POLICY_EXECUTION_RATE, ROBOT_RATE,
)
from neuracore_types import DataType, EmbodimentDescription
from neuracore.ml.preprocessing.methods.resize_pad import ResizePad

from common.data_manager import RobotActivityState
from common.policy_state import PolicyState
from common.policy_helpers import embodiment_names_ordered
from common.system_bootstrap import bootstrap_robot_system
from common.shared_actions import toggle_robot_enabled, move_robot_home
from common.robot_visualizer import RobotVisualizer
from common.threads.quest_reader import quest_reader_thread
from meta_quest_teleop.reader import MetaQuestReader
from common.policy_helpers import get_policy_embodiments, print_policy_embodiments

# Import the newly extracted policy actions
from common.policy_actions import (
    run_policy, start_policy_execution, play_policy, policy_execution_thread
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Piper Robot Policy Test")
    parser.add_argument("--ip-address", type=str, default=None)
    parser.add_argument("--no-quest", action="store_true")
    parser.add_argument("--continuous-mode", choices=["pipeline", "sequential"], default="sequential")
    
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument("--train-run-name", type=str, default=None)
    policy_group.add_argument("--model-path", type=str, default=None)
    policy_group.add_argument("--remote-endpoint-name", type=str, default=None)
    parser.add_argument("--robot-name", type=str, default="AgileX PiPER")
    args = parser.parse_args()

    print("=" * 60 + "\nPIPER ROBOT TEST WITH NEURACORE POLICY\n" + "=" * 60)

    # 1. Connect Neuracore & Define Embodiments
    nc.login()
    nc.connect_robot(robot_name=args.robot_name, urdf_path=str(URDF_PATH), overwrite=False)

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
        print(f"\n🤖 Loading policy from training run: {args.train_run_name}...")
        policy = nc.policy(
            train_run_name=args.train_run_name,
            device="cuda",
            robot_name=args.robot_name,
        )
    else:
        policy = nc.policy(
            model_file=args.model_path,
            device="cuda",
            robot_name=args.robot_name,
        )

    policy_state = PolicyState()
    
    input_emb, output_emb = get_policy_embodiments(policy)
    print_policy_embodiments(input_emb, output_emb)

    policy_state.set_execution_mode(PolicyState.ExecutionMode.TARGETING_TIME)

    # 2. Bootstrap Core System
    data_manager, robot_controller, ik_solver, active_threads = bootstrap_robot_system(start_ik=True, start_camera=True)

    # 3. Initialize Quest Reader (Conditional)
    quest_reader = None
    if not args.no_quest:
        print("\n🎮 Initializing Meta Quest reader...")
        quest_reader = MetaQuestReader(ip_address=args.ip_address, port=5555, run=True)
        quest_thread = threading.Thread(target=quest_reader_thread, args=(data_manager, quest_reader), daemon=True)
        quest_thread.start()
        active_threads.append(quest_thread)

    # 4. Initialize Visualizer & Bind Callbacks
    print("\n🖥️  Starting Viser visualization...")
    visualizer = RobotVisualizer(str(URDF_PATH))
    visualizer.add_policy_controls(PREDICTION_HORIZON_EXECUTION_RATIO, POLICY_EXECUTION_RATE, ROBOT_RATE, "targeting_time")
    visualizer.add_toggle_robot_enabled_status_button()
    visualizer.add_homing_controls()
    visualizer.add_policy_buttons()

    # Shared Button Binds
    visualizer.set_toggle_robot_enabled_status_callback(lambda: toggle_robot_enabled(data_manager, robot_controller, visualizer))
    visualizer.set_go_home_callback(lambda: move_robot_home(data_manager, robot_controller))
    if quest_reader:
        quest_reader.on("button_a_pressed", lambda: toggle_robot_enabled(data_manager, robot_controller, visualizer))
        quest_reader.on("button_b_pressed", lambda: move_robot_home(data_manager, robot_controller))

    # Policy Button Binds
    visualizer.set_run_policy_callback(lambda: run_policy(data_manager, policy, policy_state, visualizer, input_emb))
    visualizer.set_start_policy_execution_callback(lambda: start_policy_execution(data_manager, policy_state))
    visualizer.set_play_policy_callback(lambda: play_policy(data_manager, policy, policy_state, visualizer, input_emb, args.continuous_mode))
    visualizer.set_execution_mode_callback(lambda: policy_state.set_execution_mode(PolicyState.ExecutionMode(visualizer.get_execution_mode())))

    # 5. Start Policy Execution Thread
    policy_exec_thread = threading.Thread(
        target=policy_execution_thread,
        args=(policy, data_manager, policy_state, robot_controller, visualizer, input_emb),
        daemon=True
    )
    policy_exec_thread.start()
    active_threads.append(policy_exec_thread)

    print("\n🚀 Ready! Check http://localhost:8080 to visualize and run policies.\n")

    # 6. Main Loop
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        traceback.print_exc()

    # 7. Cleanup
    print("\n🧹 Cleaning up...")
    policy.disconnect()
    data_manager.request_shutdown()
    data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
    if quest_reader:
        quest_reader.stop()
    for t in active_threads:
        t.join()
    robot_controller.cleanup()
    visualizer.stop()
    nc.logout()
    print("👋 Demo stopped.")