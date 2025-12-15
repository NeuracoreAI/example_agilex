"""Quest reader thread - reads controller data and manages teleop state."""

import sys
import time
import traceback
from pathlib import Path

# Add meta_quest_teleop to path
sys.path.insert(0, str(Path(__file__).parent.parent / "meta_quest_teleop"))
from common.configs import CONTROLLER_DATA_RATE, GRIP_THRESHOLD
from common.data_manager import DataManager, RobotActivityState

from meta_quest_teleop.reader import MetaQuestReader


def quest_reader_thread(
    data_manager: DataManager, quest_reader: MetaQuestReader
) -> None:
    """Quest reader thread - reads controller data and manages teleop state.

    This thread runs at high frequency to ensure responsive controller input.
    Handles:
    - Reading Meta Quest controller data
    - Processing grip button (dead man's switch)
    - Managing teleop activation/deactivation
    - Capturing initial poses when teleop activates

    Args:
        data_manager: DataManager object for thread-safe communication
        quest_reader: MetaQuestReader instance
    """
    print("🎮 Quest Controller thread started")

    dt: float = 1.0 / CONTROLLER_DATA_RATE
    prev_grip_active: bool = False

    try:
        while not data_manager.is_shutdown_requested():
            iteration_start = time.time()

            # Get controller data
            grip_value = quest_reader.get_grip_value("right")
            trigger_value = quest_reader.get_trigger_value("right")
            controller_transform = quest_reader.get_hand_controller_transform_ros(
                hand="right"
            )

            # Update shared state with controller data
            data_manager.set_controller_data(
                controller_transform, grip_value, trigger_value
            )

            # Grip button logic (dead man's switch)
            robot_activity_state = data_manager.get_robot_activity_state()
            # Teleop can only be activated if robot is ENABLED (not HOMING or DISABLED)
            grip_active = (
                grip_value >= GRIP_THRESHOLD
                and robot_activity_state == RobotActivityState.ENABLED
            )

            # Rising edge - grip just pressed AND robot is enabled
            if (
                grip_active
                and not prev_grip_active
                and controller_transform is not None
            ):
                # Start teleop control
                # capture initial poses
                controller_initial_transform = controller_transform.copy()

                # Capture initial robot end-effector pose
                robot_initial_transform = data_manager.get_current_end_effector_pose()

                data_manager.set_teleop_state(
                    True, controller_initial_transform, robot_initial_transform
                )

                print("✓ Teleop control activated")
                print(
                    f"  Controller initial position: {controller_initial_transform[:3, 3]}"
                )
                if robot_initial_transform is not None:
                    print(f"  Robot initial position: {robot_initial_transform[:3, 3]}")
                else:
                    print("  Robot initial position: None")

            # Falling edge - grip just released OR robot disabled
            elif not grip_active and prev_grip_active:
                # Stop teleop control
                data_manager.set_teleop_state(False, None, None)
                print("✗ Teleop control deactivated")

            prev_grip_active = grip_active

            # Sleep to maintain loop rate (check shutdown more frequently)
            elapsed = time.time() - iteration_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"❌ Quest reader thread error: {e}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        # Ensure clean exit - deactivate teleop
        data_manager.set_teleop_state(False, None, None)
        print("🎮 Quest Controller thread stopped")
