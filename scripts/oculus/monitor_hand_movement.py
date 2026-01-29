#!/usr/bin/env python3
"""Monitor Meta Quest right hand controller movement and display flags for significant changes."""

import time
from collections import deque

import numpy as np
from meta_quest_teleop.reader import MetaQuestReader
from scipy.spatial.transform import Rotation


class MovementMonitor:
    """Monitor hand controller movement and detect significant changes."""

    def __init__(
        self,
        history_size: int = 5,
        position_threshold: float = 0.01,
        rotation_threshold: float = 0.05,
    ) -> None:
        """Initialize the movement monitor.

        Args:
            history_size: Number of samples to use for moving average
            position_threshold: Threshold for significant position change (meters)
            rotation_threshold: Threshold for significant rotation change (radians)
        """
        self.history_size = history_size
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold

        # History buffers
        self.position_history: deque[np.ndarray] = deque(maxlen=history_size)
        self.euler_history: deque[np.ndarray] = deque(maxlen=history_size)

        # Current values
        self.current_position = None
        self.current_euler = None

    def update(self, position: np.ndarray, euler: np.ndarray) -> None:
        """Update with new position and orientation data.

        Args:
            position: [x, y, z] position in meters
            euler: [roll, pitch, yaw] in radians
        """
        self.current_position = position
        self.current_euler = euler

        self.position_history.append(position.copy())
        self.euler_history.append(euler.copy())

    def get_movement_flags(self) -> dict:
        """Get flags indicating significant movement in each direction.

        Returns:
            Dictionary with movement flags for x+/-, y+/-, z+/-, roll+/-, pitch+/-, yaw+/-
        """
        flags = {
            "x+": False,
            "x-": False,
            "y+": False,
            "y-": False,
            "z+": False,
            "z-": False,
            "roll+": False,
            "roll-": False,
            "pitch+": False,
            "pitch-": False,
            "yaw+": False,
            "yaw-": False,
        }

        # Need at least 2 samples to detect movement
        if len(self.position_history) < 2:
            return flags

        # Calculate average of historical positions (excluding most recent)
        if len(self.position_history) > 1:
            historical_pos = np.mean(list(self.position_history)[:-1], axis=0)
            pos_delta = self.current_position - historical_pos

            # Check position changes
            if pos_delta[0] > self.position_threshold:
                flags["x+"] = True
            elif pos_delta[0] < -self.position_threshold:
                flags["x-"] = True

            if pos_delta[1] > self.position_threshold:
                flags["y+"] = True
            elif pos_delta[1] < -self.position_threshold:
                flags["y-"] = True

            if pos_delta[2] > self.position_threshold:
                flags["z+"] = True
            elif pos_delta[2] < -self.position_threshold:
                flags["z-"] = True

        # Calculate average of historical orientations (excluding most recent)
        if len(self.euler_history) > 1:
            historical_euler = np.mean(list(self.euler_history)[:-1], axis=0)
            euler_delta = self.current_euler - historical_euler

            # Wrap angle differences to [-pi, pi]
            euler_delta = np.arctan2(np.sin(euler_delta), np.cos(euler_delta))

            # Check rotation changes
            if euler_delta[0] > self.rotation_threshold:
                flags["roll+"] = True
            elif euler_delta[0] < -self.rotation_threshold:
                flags["roll-"] = True

            if euler_delta[1] > self.rotation_threshold:
                flags["pitch+"] = True
            elif euler_delta[1] < -self.rotation_threshold:
                flags["pitch-"] = True

            if euler_delta[2] > self.rotation_threshold:
                flags["yaw+"] = True
            elif euler_delta[2] < -self.rotation_threshold:
                flags["yaw-"] = True

        return flags


def format_flags(flags: dict) -> str:
    """Format movement flags as a readable string."""
    active_flags = [key for key, value in flags.items() if value]
    if not active_flags:
        return "No significant movement"
    return " ".join(active_flags)


def main() -> None:
    """Main monitoring loop."""
    print("=" * 80)
    print("Meta Quest Right Hand Pointer Monitor")
    print("=" * 80)
    print("\nConnecting to Meta Quest...")

    # Initialize reader
    try:
        reader = MetaQuestReader(ip_address=None, port=5555, run=True)  # Auto-detect
    except Exception as e:
        print(f"Error connecting to Meta Quest: {e}")
        print("Make sure the Meta Quest is connected and the APK is running.")
        return

    print("Connected! Starting monitoring...\n")

    # Initialize movement monitor
    monitor = MovementMonitor(
        history_size=10,
        position_threshold=0.05,  # 5cm
        rotation_threshold=0.1,  # ~5.75 degrees
    )

    # Main loop
    try:
        while True:
            # Update reader
            if reader.update():
                # Get right hand pointer transform
                transform = reader.get_hand_controller_transform_ros(hand="right")

                if transform is not None:
                    # Extract position
                    position = transform[:3, 3]

                    # Extract orientation as Euler angles
                    rotation = Rotation.from_matrix(transform[:3, :3])
                    euler = rotation.as_euler("xyz")  # roll, pitch, yaw

                    # Update monitor
                    monitor.update(position, euler)

                    # Get movement flags
                    flags = monitor.get_movement_flags()

                    # Clear terminal and print fresh
                    print("\033[H\033[J", end="")  # ANSI clear screen
                    print("=" * 80)
                    print("Meta Quest Right Hand Pointer Monitor")
                    print("=" * 80)
                    print()

                    # Display position
                    print(
                        f"Position (m):  x: {position[0]:7.4f}  y: {position[1]:7.4f}  z: {position[2]:7.4f}"
                    )

                    # Display orientation
                    euler_deg = np.rad2deg(euler)
                    print(
                        f"Rotation (°):  roll: {euler_deg[0]:7.2f}  pitch: {euler_deg[1]:7.2f}  yaw: {euler_deg[2]:7.2f}"
                    )

                    # Visual indicators for active movements
                    print()
                    position_indicators = []
                    if flags["x+"]:
                        position_indicators.append("X+→")
                    if flags["x-"]:
                        position_indicators.append("←X-")
                    if flags["y+"]:
                        position_indicators.append("Y+↑")
                    if flags["y-"]:
                        position_indicators.append("↓Y-")
                    if flags["z+"]:
                        position_indicators.append("Z+⊙")
                    if flags["z-"]:
                        position_indicators.append("⊗Z-")

                    rotation_indicators = []
                    if flags["roll+"]:
                        rotation_indicators.append("Roll+↻")
                    if flags["roll-"]:
                        rotation_indicators.append("↺Roll-")
                    if flags["pitch+"]:
                        rotation_indicators.append("Pitch+⤴")
                    if flags["pitch-"]:
                        rotation_indicators.append("⤵Pitch-")
                    if flags["yaw+"]:
                        rotation_indicators.append("Yaw+↷")
                    if flags["yaw-"]:
                        rotation_indicators.append("↶Yaw-")

                    if position_indicators:
                        print(f"Position:      {' '.join(position_indicators)}")
                    else:
                        print("Position:      -")

                    if rotation_indicators:
                        print(f"Rotation:      {' '.join(rotation_indicators)}")
                    else:
                        print("Rotation:      -")

                    print("-" * 80)
                    print("\nPress Ctrl+C to stop")

            # Small delay to prevent CPU spinning
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
    except Exception as e:
        print(f"\n\nError during monitoring: {e}")
        import traceback

        traceback.print_exc()

    finally:
        reader.stop()


if __name__ == "__main__":
    main()
