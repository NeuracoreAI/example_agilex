"""Leader arm reader thread – reads mapped output from leader and writes to DataManager."""

import time
import traceback

from common.data_manager import DataManager
from common.leader_arm import LerobotSO100LeaderArm


def leader_reader_thread(
    data_manager: DataManager,
    leader_arm: LerobotSO100LeaderArm,
    rate_hz: float,
) -> None:
    """Read leader arm at rate_hz via read_mapped() and store in data_manager.

    Leader arm must have configure_follower() called before starting this thread.
    """
    dt = 1.0 / rate_hz
    print("🦾 Leader reader thread started")
    try:
        while not data_manager.is_shutdown_requested():
            t0 = time.perf_counter()
            try:
                joint_angles, gripper_open = leader_arm.read_mapped()
                data_manager.set_leader_mapped_state(joint_angles, gripper_open)
            except RuntimeError as e:
                if "no calibration registered" in str(e):
                    print(
                        "❌ Leader has no calibration. Run lerobot-calibrate for the leader."
                    )
                    data_manager.request_shutdown()
                elif "configure_follower" in str(e):
                    print(
                        "❌ Leader arm has no follower config. Call configure_follower() before starting the thread."
                    )
                    data_manager.request_shutdown()
                break
            elapsed = time.perf_counter() - t0
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)
    except Exception as e:
        print(f"❌ Leader reader error: {e}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        print("🦾 Leader reader thread stopped")
