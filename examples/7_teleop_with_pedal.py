#!/usr/bin/env python3
"""Piper Robot Teleoperation with Meta Quest and Foot Pedal control."""

import argparse
import multiprocessing
import sys
import threading
import time
from pathlib import Path
import neuracore as nc

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import URDF_PATH
from common.system_bootstrap import bootstrap_robot_system
from common.shared_actions import (
    toggle_robot_enabled, move_robot_home, 
    toggle_recording, neuracore_logging_callback
)
from common.foot_pedal import FootPedal
from common.threads.quest_reader import quest_reader_thread
from meta_quest_teleop.reader import MetaQuestReader

ENABLE_DISABLE_PEDAL = "a"
HOME_POSITION_PEDAL = "b"
RECORD_TOGGLE_PEDAL = "c"

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(description="Combined Quest + Pedal Teleop")
    parser.add_argument("--ip-address", type=str, help="Meta Quest IP")
    parser.add_argument("--dataset-name", type=str, help="Neuracore dataset name")
    args = parser.parse_args()

    print("=" * 60 + "\nPIPER TELEOP: META QUEST + FOOT PEDALS\n" + "=" * 60)

    # 1. Init Neuracore
    try:
        nc.login()
        nc.connect_robot(robot_name="AgileX PiPER", urdf_path=str(URDF_PATH), overwrite=False)
        ds_name = args.dataset_name or f"pedal-teleop-{time.strftime('%H-%M-%S')}"
        nc.create_dataset(name=ds_name, description="Quest + Pedal unified collection")
    except Exception as e:
        print(f"⚠️  Neuracore init skipped/failed: {e}")

    # 2. Bootstrap Core System
    data_manager, robot_controller, ik_solver, active_threads = bootstrap_robot_system(
        start_ik=True, start_camera=True
    )
    data_manager.set_on_change_callback(neuracore_logging_callback)

    # 3. Quest Reader
    quest_reader = None
    try:
        print("🔍 Searching for Meta Quest...")
        quest_reader = MetaQuestReader(ip_address=args.ip_address, port=5555, run=True)
        quest_thread = threading.Thread(target=quest_reader_thread, args=(data_manager, quest_reader), daemon=True)
        quest_thread.start()
        active_threads.append(quest_thread)
    except (Exception, BaseException) as e:
        print(f"⚠️  Quest reader init skipped/failed: {e}")

    # 4. Foot Pedal Init & Binding
    print("\n⌨️  Initializing Foot Pedals...")
    pedal = FootPedal({
        "button_a": ENABLE_DISABLE_PEDAL,
        "button_b": HOME_POSITION_PEDAL,
        "button_c": RECORD_TOGGLE_PEDAL,
    })
    
    # 🔗 Bind the Shared Actions
    pedal.bind("button_a", lambda: toggle_robot_enabled(data_manager, robot_controller))
    pedal.bind("button_b", lambda: move_robot_home(data_manager, robot_controller))
    pedal.bind("button_c", lambda: toggle_recording(play_audio=False))

    pedal_thread = threading.Thread(target=pedal.run_loop, args=(data_manager,), daemon=True)
    pedal_thread.start()
    active_threads.append(pedal_thread)

    print("\n✅ SYSTEM ONLINE")
    print("🎮 QUEST CONTROLS: Hold GRIP to move, TRIGGER for gripper")
    print("⌨️  PEDAL CONTROLS: ENABLE/DISABLE (A), HOME (B), RECORD (C)")

    # 5. Wait Loop
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        if nc.is_recording(): nc.cancel_recording()
        try: nc.logout()
        except: pass
        data_manager.request_shutdown()
        if quest_reader: quest_reader.stop()
        for t in active_threads: t.join(timeout=1.0)
        if robot_controller: robot_controller.cleanup()