#!/usr/bin/env python3
"""
Piper Robot Teleoperation with Meta Quest and Foot Pedals.

This script provides a unified control interface for the AgileX Piper robot,
combining the 6D spatial tracking of a Meta Quest controller with the 
hands-free convenience of a USB foot pedal. It simultaneously streams 
synchronized telemetry data to the Neuracore cloud.

Key Features:
    - Meta Quest for precise 6-DoF end-effector manipulation.
    - Foot Pedals mapped to system states (Enable/Disable, Home, Record).
    - Dynamically loads IK and scaling configurations from a YAML file.
    - Robust ADB connection handling for the VR headset.

Hardware Requirements:
    - AgileX Piper robot arm connected via CAN interface ('can0').
    - Meta Quest headset connected via USB with 'USB Debugging' explicitly allowed.
    - 3-Button USB Foot Pedal (mapped to keystrokes 'a', 'b', 'c').

Usage:
    python 7_teleop_with_pedal.py --ip-address <QUEST_IP> --ik-config ik_conf/default.yaml
"""

import argparse
import multiprocessing
import sys
import threading
import time
import traceback
from pathlib import Path

import neuracore as nc

# ---------------------------------------------------------------------------
# Path Configuration & Local Imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import URDF_PATH, META_QUEST_AXIS_MASK
from common.config_parser import load_ik_config
from common.data_manager import RobotActivityState
from common.system_bootstrap import bootstrap_robot_system
from common.shared_actions import (
    toggle_robot_enabled, move_robot_home, 
    toggle_recording, neuracore_logging_callback
)
from common.foot_pedal import FootPedal
from common.threads.quest_reader import quest_reader_thread
from meta_quest_teleop.reader import MetaQuestReader

# Hardware key mappings for the USB foot pedal
ENABLE_DISABLE_PEDAL = "a"
HOME_POSITION_PEDAL = "b"
RECORD_TOGGLE_PEDAL = "c"

if __name__ == "__main__":
    # Ensure safe multiprocess spawning for UI/Background threads
    multiprocessing.set_start_method("spawn")
    
    # ---------------------------------------------------------
    # 1. Argument Parsing
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Combined Quest + Pedal Teleop")
    parser.add_argument("--ip-address", type=str, default=None, help="Meta Quest IP address on the local network.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Override the auto-generated dataset name.")
    parser.add_argument("--ik-config", type=str, default="ik_conf/default.yaml", help="Path to IK/teleop YAML config.")
    args = parser.parse_args()

    print("=" * 60 + "\nPIPER TELEOP: META QUEST + FOOT PEDALS\n" + "=" * 60)

    # Load the YAML configuration dictionary
    config = load_ik_config(args.ik_config)

    # ---------------------------------------------------------
    # 2. Neuracore Initialization & Dataset Creation
    # ---------------------------------------------------------
    print("\n🔧 Initializing Neuracore...")
    try:
        nc.login()
        nc.connect_robot(robot_name="AgileX PiPER", urdf_path=str(URDF_PATH), overwrite=False)
        
        ds_name = args.dataset_name or f"pedal-teleop-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        nc.create_dataset(name=ds_name, description="Quest + Pedal unified teleop collection")
    except Exception as e:
        print(f"⚠️  Neuracore initialization skipped/failed: {e}")

    # ---------------------------------------------------------
    # 3. Hardware & Subsystem Bootstrapping
    # ---------------------------------------------------------
    data_manager, robot_controller, ik_solver, active_threads = bootstrap_robot_system(
        config, start_ik=True, start_camera=True
    )
    
    # Wire Neuracore logging to DataManager so states push to the cloud
    data_manager.set_on_change_callback(neuracore_logging_callback)

    # ---------------------------------------------------------
    # 4. Meta Quest Initialization & Error Handling
    # ---------------------------------------------------------
    print("\n🎮 Initializing Meta Quest reader...")
    quest_reader = None
    try:
        quest_reader = MetaQuestReader(
            ip_address=args.ip_address, port=5555, run=True, axis_mask=META_QUEST_AXIS_MASK
        )
        quest_thread = threading.Thread(target=quest_reader_thread, args=(data_manager, quest_reader), daemon=True)
        quest_thread.start()
        active_threads.append(quest_thread)
    except (Exception, SystemExit) as e:
        print("\n" + "!" * 60)
        print("❌ FAILED TO ACCESS META QUEST")
        print("!" * 60)
        print("The headset is plugged in, but ADB debugging permissions are missing.")
        print("\nPLEASE FOLLOW THESE STEPS:")
        print("  1. Put the Meta Quest headset on your head.")
        print("  2. Look for a notification in your menu that says 'USB Detected'.")
        print("  3. Click on that notification and select 'Allow' to grant data access.")
        print("  4. Rerun this script.")
        print("!" * 60 + "\n")
        
        data_manager.request_shutdown()
        for t in active_threads:
            t.join(timeout=1.0)
        if robot_controller:
            robot_controller.cleanup()
        sys.exit(1)

    # ---------------------------------------------------------
    # 5. Foot Pedal Initialization & Binding
    # ---------------------------------------------------------
    print("\n⌨️  Initializing Foot Pedals...")
    pedal = FootPedal({
        "button_a": ENABLE_DISABLE_PEDAL,
        "button_b": HOME_POSITION_PEDAL,
        "button_c": RECORD_TOGGLE_PEDAL,
    })
    
    # Bind the hardware actions to the pedal inputs
    pedal.bind("button_a", lambda: toggle_robot_enabled(data_manager, robot_controller))
    pedal.bind("button_b", lambda: move_robot_home(data_manager, robot_controller))
    pedal.bind("button_c", lambda: toggle_recording(play_audio=False))

    pedal_thread = threading.Thread(target=pedal.run_loop, args=(data_manager,), daemon=True)
    pedal_thread.start()
    active_threads.append(pedal_thread)

    print("\n✅ SYSTEM ONLINE")
    print("🎮 QUEST CONTROLS: Hold GRIP to move arm, TRIGGER to close gripper")
    print("⌨️  PEDAL CONTROLS: ENABLE/DISABLE (Left), HOME (Middle), RECORD (Right)\n")

    # ---------------------------------------------------------
    # 6. Main Daemon Loop
    # ---------------------------------------------------------
    try:
        # Keep the main thread alive while background threads handle execution
        while not data_manager.is_shutdown_requested():
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n👋 Interrupt received - shutting down gracefully...")
        data_manager.request_shutdown()
    except Exception as e:
        print(f"\n❌ Unhandled Demo error: {e}")
        traceback.print_exc()
        data_manager.request_shutdown()

    # ---------------------------------------------------------
    # 7. Graceful Teardown & Cleanup
    # ---------------------------------------------------------
    finally:
        print("\n🧹 Cleaning up subsystems...")
        
        # Ensure we don't leave dangling, unterminated recordings on the cloud
        if nc.is_recording(): 
            nc.cancel_recording()
            
        try:
            nc.logout()
        except Exception: 
            pass
            
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        data_manager.request_shutdown()
        
        if quest_reader: 
            quest_reader.stop()
            
        for t in active_threads: 
            t.join(timeout=2.0)
            
        if robot_controller: 
            robot_controller.cleanup()
            
        print("👋 Demo stopped.")