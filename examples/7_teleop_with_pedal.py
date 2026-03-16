#!/usr/bin/env python3
"""Piper Robot Teleoperation with Meta Quest and Foot Pedal control.

This script combines Meta Quest controller input for movement with Foot Pedal
control for session management (Activate, Home, Record).
"""

import argparse
import multiprocessing
import sys
import threading
import time
from pathlib import Path
from typing import Any

import neuracore as nc
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add neuracore path for local imports if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "neuracore"))

import json
import traceback
from typing import Callable

from common.configs import (
    CONTROLLER_BETA,
    CONTROLLER_D_CUTOFF,
    CONTROLLER_MIN_CUTOFF,
    GRIPPER_FRAME_NAME,
    GRIPPER_LOGGING_NAME,
    JOINT_NAMES,
    NEUTRAL_JOINT_ANGLES,
    POSTURE_COST_VECTOR,
    ROBOT_RATE,
    URDF_PATH,
)
from common.data_manager import DataManager, RobotActivityState
from common.threads.camera import camera_thread
from common.threads.ik_solver import ik_solver_thread
from common.threads.joint_state import joint_state_thread
from common.threads.quest_reader import quest_reader_thread
from meta_quest_teleop.reader import MetaQuestReader

from pink_ik_solver import PinkIKSolver
from piper_controller import PiperController

# ---------------------------------------------------------------------------
# Foot Pedal logic (Moved here per PR review)
# ---------------------------------------------------------------------------

_PEDAL_CONFIG_PATH = Path.home() / ".neuracore" / "foot_pedal.json"
_DEFAULT_MAPPINGS = {"button_a": "a", "button_b": "b", "button_c": "c"}


class FootPedal:
    """Foot pedal reader class - fires callbacks on key press."""

    def __init__(self, data_manager: DataManager, config: dict[str, Any] | None = None):
        """Initialize FootPedal with data_manager and optional config."""
        self._data_manager = data_manager
        self._config = config or self.load_config()
        self._mappings = {
            "button_a": self._config.get("button_a"),
            "button_b": self._config.get("button_b"),
            "button_c": self._config.get("button_c"),
        }

        # Callbacks
        self.on_button_a: Callable[[], None] | None = None
        self.on_button_b: Callable[[], None] | None = None
        self.on_button_c: Callable[[], None] | None = None

    @staticmethod
    def load_config() -> dict[str, Any]:
        """Load foot pedal key mappings from JSON file."""
        if _PEDAL_CONFIG_PATH.exists():
            try:
                with open(_PEDAL_CONFIG_PATH) as f:
                    return dict(json.load(f))
            except Exception as e:
                print(f"⚠️  Could not load pedal config: {e}")
        return dict(_DEFAULT_MAPPINGS)

    def _dispatch(self, char: str) -> None:
        """Fire the right callback for the detected key char."""
        if char == self._mappings.get("button_a") and self.on_button_a:
            self.on_button_a()
        elif char == self._mappings.get("button_b") and self.on_button_b:
            self.on_button_b()
        elif char == self._mappings.get("button_c") and self.on_button_c:
            self.on_button_c()

    def run(self) -> None:
        """Main listener loop."""
        print(f"⌨️  Foot pedal listener started. Mappings: {self._mappings}")

        # -- evdev path (preferred on Linux) ------------------------------------
        try:
            import evdev

            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            pedals = [
                d for d in devices if "PCsensor" in d.name or "FootSwitch" in d.name
            ]

            if pedals:
                pedal_dev = pedals[0]
                for p in pedals:
                    if "Keyboard" in p.name:
                        pedal_dev = p
                        break

                print(f"⌨️  Foot pedal acquired via evdev: {pedal_dev.name}")
                try:
                    pedal_dev.grab()
                    for event in pedal_dev.read_loop():
                        if self._data_manager.is_shutdown_requested():
                            break
                        if event.type == evdev.ecodes.EV_KEY:
                            k = evdev.categorize(event)
                            if k.keystate == k.key_down:
                                key_str = k.keycode
                                if isinstance(key_str, list):
                                    key_str = key_str[0]
                                char = key_str.replace("KEY_", "").lower()
                                print(f"🔍 [PEDAL] Key: '{char}'")
                                self._dispatch(char)
                except Exception as e:
                    print(f"⚠️  evdev read error: {e}")
                finally:
                    try:
                        pedal_dev.ungrab()
                    except Exception:
                        pass
                print("⌨️  Foot pedal thread stopped (evdev).")
                return

        except Exception as e:
            print(f"⚠️  evdev unavailable: {e} — falling back to pynput")

        # -- pynput fallback ----------------------------------------------------
        try:
            from pynput import keyboard

            print("⌨️  Foot pedal listener (pynput fallback) started.")

            def on_press(key: object) -> None:
                try:
                    char = key.char if hasattr(key, "char") else str(key)
                    self._dispatch(char)
                except Exception:
                    pass

            with keyboard.Listener(on_press=on_press) as listener:
                while not self._data_manager.is_shutdown_requested():
                    if not listener.is_alive():
                        break
                    time.sleep(0.1)
                listener.stop()

        except Exception as e:
            print(f"✗ Fatal error in foot pedal: {e}")
            traceback.print_exc()
        finally:
            print("⌨️  Foot pedal listener stopped.")


def log_to_neuracore_on_change_callback(
    name: str, value: Any, timestamp: float
) -> None:
    """Callback triggered on state changes to log data to Neuracore.

    Args:
        name: Name of the data stream.
        value: Data value (float, array, or image).
        timestamp: Time of the change.
    """
    try:
        if name == "log_joint_positions":
            data_dict = {jn: np.radians(a) for jn, a in zip(JOINT_NAMES, value)}
            nc.log_joint_positions(data_dict, timestamp=timestamp)
        elif name == "log_joint_target_positions":
            data_dict = {jn: np.radians(a) for jn, a in zip(JOINT_NAMES, value)}
            nc.log_joint_target_positions(data_dict, timestamp=timestamp)
        elif name == "log_parallel_gripper_open_amounts":
            nc.log_parallel_gripper_open_amounts(
                {GRIPPER_LOGGING_NAME: value}, timestamp=timestamp
            )
        elif name == "log_parallel_gripper_target_open_amounts":
            nc.log_parallel_gripper_target_open_amounts(
                {GRIPPER_LOGGING_NAME: value}, timestamp=timestamp
            )
        elif name == "log_rgb":
            nc.log_rgb("rgb", value, timestamp=timestamp)
    except Exception as e:
        print(f"⚠️  Logging failed: {e}")


def toggle_robot_state() -> None:
    """Toggle the robot's activity state between ENABLED and DISABLED."""
    state = data_manager.get_robot_activity_state()
    if state == RobotActivityState.ENABLED:
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        robot_controller.graceful_stop()
        data_manager.set_teleop_state(False, None, None)
        print("✓ [ACTION] Robot DISABLED")
    elif state == RobotActivityState.DISABLED:
        if robot_controller.resume_robot():
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
            print("✓ [ACTION] Robot ENABLED")
        else:
            print("✗ [ACTION] Failed to enable robot")


def move_robot_home() -> None:
    """Command the robot to move to its neutral/home position."""
    state = data_manager.get_robot_activity_state()
    if state == RobotActivityState.ENABLED:
        print("🏠 [ACTION] Moving to home position...")
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
        data_manager.set_teleop_state(False, None, None)
        if not robot_controller.move_to_home():
            print("✗ [ACTION] Homing failed")
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
    else:
        print("⚠️  [ACTION] Cannot home: robot not enabled")


def toggle_recording() -> None:
    """Start or stop a data recording session in Neuracore."""
    if not nc.is_recording():
        try:
            nc.start_recording()
            print("✓ [ACTION] Recording STARTED")
        except Exception as e:
            print(f"✗ [ACTION] Recording failed to start: {e}")
    else:
        try:
            nc.stop_recording()
            print("✓ [ACTION] Recording STOPPED")
        except Exception as e:
            print(f"✗ [ACTION] Recording failed to stop: {e}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Combined Quest + Pedal Teleoperation")
    parser.add_argument("--ip-address", type=str, help="Meta Quest IP")
    parser.add_argument("--dataset-name", type=str, help="Neuracore dataset name")
    args = parser.parse_args()

    print("=" * 60)
    print("PIPER TELEOP: META QUEST + FOOT PEDALS")
    print("=" * 60)

    # Neuracore Init
    print("\n🔧 Initializing Neuracore...")
    nc.login()
    nc.connect_robot(
        robot_name="AgileX PiPER", urdf_path=str(URDF_PATH), overwrite=False
    )

    ds_name = args.dataset_name or f"pedal-teleop-{time.strftime('%H-%M-%S')}"
    nc.create_dataset(name=ds_name, description="Quest + Pedal unified collection")

    # Shared State
    data_manager = DataManager()
    data_manager.set_on_change_callback(log_to_neuracore_on_change_callback)
    data_manager.set_controller_filter_params(
        CONTROLLER_MIN_CUTOFF, CONTROLLER_BETA, CONTROLLER_D_CUTOFF
    )

    # Robot Initialization
    print("\n🤖 Initializing Piper...")
    robot_controller = PiperController(can_interface="can0", robot_rate=ROBOT_RATE)
    robot_controller.start_control_loop()

    # Threads
    print("\n📊 Starting Threads (JointState, QuestReader, IKSolver, Camera)...")
    threading.Thread(
        target=joint_state_thread, args=(data_manager, robot_controller), daemon=True
    ).start()

    quest_reader = MetaQuestReader(ip_address=args.ip_address, port=5555, run=True)
    threading.Thread(
        target=quest_reader_thread, args=(data_manager, quest_reader), daemon=True
    ).start()

    # Sync IK solver to current position
    initial_joints = np.radians(
        data_manager.get_current_joint_angles() or NEUTRAL_JOINT_ANGLES
    )
    ik_solver = PinkIKSolver(
        urdf_path=URDF_PATH,
        end_effector_frame=GRIPPER_FRAME_NAME,
        initial_configuration=initial_joints,
        posture_cost_vector=np.array(POSTURE_COST_VECTOR),
    )
    threading.Thread(
        target=ik_solver_thread, args=(data_manager, ik_solver), daemon=True
    ).start()
    threading.Thread(target=camera_thread, args=(data_manager,), daemon=True).start()

    # Foot Pedal – started as a daemon thread, callbacks wired inline
    print("\n⌨️  Initializing Foot Pedals...")
    pedal = FootPedal(data_manager)
    pedal.on_button_a = toggle_robot_state
    pedal.on_button_b = move_robot_home
    pedal.on_button_c = toggle_recording

    pedal_thread = threading.Thread(target=pedal.run, daemon=True)
    pedal_thread.start()

    print("\n✅ SYSTEM ONLINE")
    print("------------------------------------------------------------")
    print("🎮 QUEST CONTROLS:   Hold GRIP to move, TRIGGER for gripper")
    print("⌨️  PEDAL CONTROLS:   ACTIVATE (Enable), HOME (Reset), RECORD")
    print("------------------------------------------------------------")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        pedal_thread.join(timeout=1.0)
        if nc.is_recording():
            nc.cancel_recording()
        nc.logout()
        data_manager.request_shutdown()
        quest_reader.stop()
        robot_controller.cleanup()
