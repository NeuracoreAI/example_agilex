#!/usr/bin/env python3
"""Example of using the Foot Pedal to control the Piper robot."""

import sys
import threading
import time
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import neuracore as nc
from neuracore.core.input_devices.foot_pedal import FootPedal
from piper_controller import PiperController


def main():
    print("=" * 60)
    print("FOOT PEDAL ROBOT CONTROL DEMO")
    print("=" * 60)

    # Initialize robot
    print("\n🤖 Initializing Piper robot...")
    try:
        robot = PiperController(can_interface="can0", debug_mode=False)
        robot.start_control_loop()
    except Exception as e:
        print(f"✗ Failed to initialize robot: {e}")
        print("Running in simulation/mock mode (no hardware)...")
        robot = None

    # Initialize Foot Pedal
    print("\n⌨️  Initializing Foot Pedal...")
    pedal = FootPedal()
    
    if not any(pedal.mappings.values()):
        print("⚠️  Foot pedal mappings are empty. Please run run_pedal_config.sh first.")
        # Default fallback for demo
        pedal.mappings = {"activate": "1", "home": "2", "record": "3"}
        print(f"Using default demo mappings: {pedal.mappings}")

    # Connect to Neuracore (optional, for recording)
    try:
        nc.login()
        print("✓ Connected to Neuracore")
    except Exception:
        print("⚠️  Neuracore connection failed. Recording actions will only print messages.")

    # Define callbacks
    def on_activate():
        print("\n🚀 Foot Pedal: ACTIVATE pressed")
        if robot:
            if robot.resume_robot():
                print("✓ Robot enabled")
            else:
                print("✗ Failed to enable robot")
        else:
            print("Action: Robot ACTIVATE (MOCK)")

    def on_home():
        print("\n🏠 Foot Pedal: HOME pressed")
        if robot:
            robot.move_to_home()
            print("✓ Robot moving to home")
        else:
            print("Action: Robot HOME (MOCK)")

    def on_record():
        print("\n📹 Foot Pedal: RECORD pressed")
        try:
            if not nc.is_recording():
                nc.start_recording()
                print("✓ Recording started")
            else:
                nc.stop_recording()
                print("✓ Recording stopped")
        except Exception as e:
            print(f"⚠️ Recording action failed: {e}")
            print(f"Action: Toggle Recording (MOCK)")

    # Register callbacks
    pedal.on("activate", on_activate)
    pedal.on("home", on_home)
    pedal.on("record", on_record)

    # Start pedal listener
    pedal.start()

    print("\n✅ Setup complete!")
    print(f"ACTIVATE: Press '{pedal.mappings['activate']}' to enable robot")
    print(f"HOME:     Press '{pedal.mappings['home']}' to send robot home")
    print(f"RECORD:   Press '{pedal.mappings['record']}' to toggle recording")
    print("\n⚠️  Press Ctrl+C to exit")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Stopping...")
    finally:
        pedal.stop()
        if robot:
            robot.cleanup()


if __name__ == "__main__":
    main()
