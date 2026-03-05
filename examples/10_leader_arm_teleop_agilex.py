#!/usr/bin/env python3
"""AgileX PiPER Teleoperation using LeRobot Leader Arm and Foot Pedal.

This script maps a physical LeRobot leader arm and a 3-button Foot Pedal 
to a simulated AgileX PiPER robot in Neuracore.
"""

import argparse
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

# Resolve workspace paths
_file_path = Path(__file__).resolve()
_example_agilex_root = _file_path.parent.parent
_ws_root = _example_agilex_root.parent

# Add example_so101 to path for the leader_arm module
_example_so101_root = _ws_root / "example_so101"
sys.path.insert(0, str(_example_so101_root))
sys.path.insert(0, str(_example_so101_root / "examples"))

# Add neuracore and other dependencies
_neuracore_pkg_root = _ws_root / "neuracore"
sys.path.insert(0, str(_neuracore_pkg_root))

# CRITICAL: Set PYTHONPATH so that spawned subprocesses (like the data daemon) can find neuracore
os.environ["PYTHONPATH"] = str(_neuracore_pkg_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

import neuracore as nc

# Import from example_so101 common components
from common.data_manager import DataManager, RobotActivityState
from common.leader_arm import LerobotSO101LeaderArm
from neuracore.core.input_devices.foot_pedal import FootPedal

# AgileX PiPER Configurations
PIPER_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
# Limits in degrees
PIPER_LIMITS_DEG = np.array([
    (-150.0, 150.0), (0.0, 180.0), (-170.0, 0.0),
    (-100.0, 100.0), (-70.0, 70.0), (-120.0, 120.0)
], dtype=np.float64)

PIPER_OFFSETS_DEG = np.zeros(6, dtype=np.float64)
PIPER_DIRECTIONS = np.ones(6, dtype=np.float64)

# Leader (5 joints) to AgileX (6 joints) mapping
# Leader 0 (Pan) -> Piper joint1
# Leader 1 (Lift) -> Piper joint2
# Leader 2 (Elbow) -> Piper joint3
# Leader 3 (Wrist Flex) -> Piper joint5
# Leader 4 (Wrist Roll) -> Piper joint6
LEADER_TO_PIPER_JOINT = {0: 0, 1: 1, 2: 2, 3: 4, 4: 5}
PIPER_FIXED_JOINTS = {3: 0.0} # joint4 fixed at 0

URDF_PATH = _example_agilex_root / "piper_description" / "urdf" / "piper_description.urdf"

def leader_reader_thread(data_manager: DataManager, leader: LerobotSO101LeaderArm, rate: float):
    dt = 1.0 / rate
    while not data_manager.is_shutdown_requested():
        try:
            if leader.is_connected:
                angles, gripper = leader.read_mapped()
                data_manager.set_leader_mapped_state(angles, gripper)
        except Exception as e:
            print(f"Error in leader reader thread: {e}")
        time.sleep(dt)

def main():
    parser = argparse.ArgumentParser(description="AgileX PiPER Teleop with LeRobot Leader and Foot Pedal")
    parser.add_argument("--leader-port", type=str, default="/dev/ttyACM0", help="USB port for leader arm")
    parser.add_argument("--leader-id", type=str, default="my_awesome_leader_arm", help="LeRobot calibration id")
    parser.add_argument("--leader-rate", type=float, default=50.0, help="Reading rate in Hz")
    parser.add_argument("--robot-name", type=str, default="AgileX PiPER", help="Robot name in Neuracore")
    
    # Pedal keys
    parser.add_argument("--pedal-activate", type=str, default="a")
    parser.add_argument("--pedal-home", type=str, default="b")
    parser.add_argument("--pedal-record", type=str, default="c")
    
    args = parser.parse_args()

    print("=" * 60)
    print("AGILEX PIPER LE ROBOT + FOOT PEDAL TELEOP")
    print("=" * 60)

    # 1. Initialize Leader Arm
    print(f"\n🦾 Connecting to Le Robot leader on {args.leader_port}...")
    leader = LerobotSO101LeaderArm(port=args.leader_port, calibration_id=args.leader_id)
    try:
        leader.connect(calibrate=False)
    except Exception as e:
        print(f"❌ Failed to connect to leader: {e}")
        print("Tip: Run the robust calibration script first.")
        sys.exit(1)
    
    # Adjusted Piper Config for better mapping
    local_limits_deg = np.array([
        (-150.0, 150.0), (0.0, 160.0), (-160.0, 0.0), # J1, J2, J3
        (-100.0, 100.0), (-100.0, 100.0), (-180.0, 180.0) # J4, J5, J6
    ], dtype=np.float64)
    
    # Official Neutral Angles: [-1.0, 80.0, -51.0, -4.0, 16.0, 2.6]
    # We use these as offsets so leader 0 -> official neutral
    local_offsets_deg = np.array([-1.0, 80.0, -51.0, 0.0, 16.0, 2.6])
    
    # Directions: 
    # J3 (Elbow) often needs to be flipped if leader fold -> piper fold
    # J2 (Lift) leader up -> piper lift up
    local_directions = np.array([1.0, 1.0, -1.0, 1.0, 1.0, 1.0])

    leader.configure_follower(
        follower_limits_deg=local_limits_deg,
        follower_offsets_deg=local_offsets_deg,
        follower_directions=local_directions,
        leader_to_follower_joint=LEADER_TO_PIPER_JOINT,
        fixed_joints=PIPER_FIXED_JOINTS
    )
    print("✓ Leader arm configured with OFFICIAL offsets")

    # Neutral Pose for Parking/Home (degrees) - Using official values
    NEUTRAL_POSE_DEG = np.array([-1.0, 80.0, -51.0, 0.0, 16.0, 2.6])

    # 2. Initialize Data Manager and Foot Pedal
    data_manager = DataManager()
    data_manager.set_robot_activity_state(RobotActivityState.DISABLED) # Start DISABLED
    
    pedal = FootPedal()
    pedal.save_config(activate_key=args.pedal_activate, home_key=args.pedal_home, record_key=args.pedal_record)

    # Define callbacks first
    def on_activate():
        state = data_manager.get_robot_activity_state()
        if state == RobotActivityState.ENABLED:
            data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
            print("✓ [PEDAL] Robot DISABLED (Parking...)")
        else:
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
            print("✓ [PEDAL] Robot ENABLED (Following Leader)")

    def on_home():
        print("🏠 [PEDAL] Moving to home position...")
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)

    def on_record():
        print("⏺️ [PEDAL] Toggling recording...")
        if not nc.is_recording():
            try:
                nc.start_recording()
                print("✓ [PEDAL] Recording STARTED")
            except Exception as e:
                print(f"✗ [PEDAL] Failed to start recording: {e}")
        else:
            try:
                nc.stop_recording()
                print("✓ [PEDAL] Recording STOPPED")
            except Exception as e:
                print(f"✗ [PEDAL] Failed to stop recording: {e}")

    # Explicitly register
    pedal.on("activate", on_activate)
    pedal.on("home", on_home)
    pedal.on("record", on_record)

    pedal.start()
    print(f"⌨️  Foot Pedal listener started (Keys: {args.pedal_activate}, {args.pedal_home}, {args.pedal_record})")

    # 3. Initialize Neuracore
    print("\n🔧 Initializing Neuracore...")
    nc.login()
    print(f"📦 Connecting to Robot: {args.robot_name}")
    nc.connect_robot(robot_name=args.robot_name, urdf_path=str(URDF_PATH))
    
    ds_name = f"piper-leader-teleop-{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"📂 Creating Dataset: {ds_name}")
    nc.create_dataset(name=ds_name)

    # 4. Start Leader Reading Thread
    reader_thread = threading.Thread(
        target=leader_reader_thread,
        args=(data_manager, leader, args.leader_rate),
        daemon=True
    )
    reader_thread.start()

    print("\n🚀 TELEOP READY")
    print("------------------------------------------------------------")
    print("1. Robot starts in DISABLED (Neutral) pose.")
    print(f"2. Press '{args.pedal_activate}' to ENABLE (Mirror Leader).")
    print(f"3. Press '{args.pedal_record}' to Toggle Recording.")
    print("------------------------------------------------------------")
    
    loop_count = 0
    try:
        while True:
            t0 = time.time()
            mapped_angles, mapped_gripper = data_manager.get_leader_mapped_state()
            state = data_manager.get_robot_activity_state()
            
            if state == RobotActivityState.HOMING:
                angles_deg = NEUTRAL_POSE_DEG
                gripper_val = 0.5
                time.sleep(0.5)
                data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
            elif state == RobotActivityState.DISABLED:
                angles_deg = NEUTRAL_POSE_DEG
                gripper_val = 0.5
            else:
                # ENABLED: Follow Leader
                if mapped_angles is not None:
                    angles_deg = mapped_angles
                    gripper_val = mapped_gripper
                else:
                    angles_deg = NEUTRAL_POSE_DEG
                    gripper_val = 0.5

            # Streaming and Logging
            if angles_deg is not None:
                loop_count += 1
                # Debug every 30 loops (~2Hz)
                if loop_count % 30 == 0:
                    if state == RobotActivityState.DISABLED:
                        print(f"💡 [REMINDER] Robot is DISABLED. Press '{args.pedal_activate}' to Mirror. [Raw Grip: {gripper_val:.2f}]")
                    else:
                        print(f"📡 [MIRRORING] J1-3: {np.round(angles_deg[:3], 1)} | Grip: {gripper_val:.2f}")

                # Log to Neuracore
                data_dict = {}
                for i, deg in enumerate(angles_deg):
                    val = float(np.radians(deg))
                    # Explicit small offset for joint4 if 0.0 is problematic
                    if i == 3 and abs(val) < 0.0001:
                        val = 0.0001
                    data_dict[f"joint{i+1}"] = val
                
                # Visual Gripper Joints (joint7, joint8)
                if gripper_val is not None:
                    # joint7: 0 (closed) to 0.035 (open)
                    # joint8: 0 (closed) to -0.035 (open)
                    data_dict["joint7"] = float(gripper_val * 0.035)
                    data_dict["joint8"] = float(-gripper_val * 0.035)

                nc.log_joint_positions(data_dict, timestamp=t0)
                
            else:
                if loop_count % 60 == 0:
                    print("⚠️  [WARNING] No leader data received! Is it plugged in?")
                
            time.sleep(max(0, 1.0 / 60.0 - (time.time() - t0)))

    except KeyboardInterrupt:
        print("\n👋 Teleop stopped by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
    finally:
        data_manager.request_shutdown()
        pedal.stop()
        reader_thread.join(timeout=1.0)
        leader.disconnect()
        nc.logout()
        print("👋 Shutdown complete.")

if __name__ == "__main__":
    main()
