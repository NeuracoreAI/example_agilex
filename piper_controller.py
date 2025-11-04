#!/usr/bin/env python3
"""Core Piper Robot Controller."""

import threading
import time
from enum import Enum
from typing import Any

from piper_sdk import C_PiperInterface_V2  # type: ignore[attr-defined]


class PiperController:
    """Core PiPER robot controller that manages robot state and communication.

    This class is independent of any UI framework and can be used with
    GUI, VR, or other control interfaces through a command queue.
    """

    class ControlMode(Enum):
        """Control mode enumeration."""

        END_EFFECTOR = "end_effector"
        JOINT_SPACE = "joint_space"

    def __init__(
        self,
        can_interface: str = "can0",
        robot_rate: float = 100.0,
        control_mode: "PiperController.ControlMode" = ControlMode.END_EFFECTOR,
        debug_mode: bool = False,
    ) -> None:
        """Initialize the robot controller.

        Args:
            can_interface: CAN interface for robot communication (default: 'can0')
            robot_rate: Robot control loop rate in Hz (default: 100.0)
            control_mode: Initial control mode (END_EFFECTOR or JOINT_SPACE)
            debug_mode: Enable debug logging (default: False)
        """
        self.can_interface = can_interface
        self.robot_rate = robot_rate
        self.debug_mode = debug_mode

        # Thread synchronization
        self.position_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.running = threading.Event()
        self.running.set()

        # Robot operational state
        self._robot_enabled = False

        self._control_loop_thread = threading.Thread(
            target=self.control_loop, daemon=True
        )

        # Control mode
        self._control_mode = control_mode

        # HOME positions in end effector space and joint space

        # NOTE: this is set to a preferred home pose for the robot
        self.HOME_POSE = [-3.123, -125.085, 382.251, -78.132, 84.303, -169.496]
        self.HOME_JOINT_ANGLES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.HOME_GRIPPER_OPEN_VALUE = 53.800

        # Joint limits from SDK documentation (in degrees)
        self.JOINT_LIMITS = [
            [-150.0, 150.0],  # Joint 1
            [0.0, 180.0],  # Joint 2
            [-170.0, 0.0],  # Joint 3
            [-100.0, 100.0],  # Joint 4
            [-70.0, 70.0],  # Joint 5
            [-120.0, 120.0],  # Joint 6
        ]

        self.GRIPPER_LIMITS = [1.000, 101.00]  # in degrees

        # End-effector target pose
        self._target_pose = self.HOME_POSE.copy()

        # Joint space target angles
        self._target_joint_angles = self.HOME_JOINT_ANGLES.copy()

        # Gripper target open value
        self._gripper_open_value = self.HOME_GRIPPER_OPEN_VALUE

        # Initialize robot connection
        self._initialize_robot()

        if self.debug_mode:
            print(f"PiperController initialized at {robot_rate} Hz")

    def start_control_loop(self) -> None:
        """Start the control loop thread."""
        self._control_loop_thread.start()

    def stop_control_loop(self) -> None:
        """Stop the control loop thread."""
        self.running.clear()
        if self._control_loop_thread.is_alive():
            self._control_loop_thread.join()
            print("✓ Control loop thread joined")
        else:
            print("Control loop thread is not running")

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources and disconnect from robot."""
        print("🧹 Cleaning up robot controller...")
        self.stop_control_loop()

        if (
            hasattr(self, "piper")
            and self.piper is not None
            and self.piper.get_connect_status()
        ):
            self.graceful_stop()  # this will disable the robot as well
            self.piper.DisconnectPort()
            print("✓ Robot disconnected")

        print("✓ Robot controller cleanup completed")

    def is_robot_enabled(self) -> bool:
        """Check if robot is enabled and ready to accept commands.

        Returns:
            True if robot is enabled, False otherwise
        """
        with self.state_lock:
            return self._robot_enabled

    def _set_robot_status_enabled(self, enabled: bool) -> None:
        """Set the robot status enabled state (internal method).

        Args:
            enabled: True to enable robot, False to disable
        """
        with self.state_lock:
            old_state = self._robot_enabled
            self._robot_enabled = enabled
            if self.debug_mode:
                print(f"Robot enabled state changed: {old_state} -> {enabled}")

    def _initialize_robot(self) -> None:
        """Initialize robot connection and enable it."""
        print(f"Initializing robot on {self.can_interface}...")
        self.piper = C_PiperInterface_V2(
            self.can_interface, start_sdk_joint_limit=True, start_sdk_gripper_limit=True
        )
        self.piper.ConnectPort()

        self._enable_robot()

        print("✓ Robot initialized successfully!")

    def _enable_robot(self) -> None:
        """Enable the robot."""
        while not self.piper.EnablePiper():
            time.sleep(0.01)

        self._set_robot_status_enabled(True)
        print("✓ Robot enabled successfully!")

    def _disable_robot(self) -> None:
        """Disable the robot."""
        while self.piper.DisablePiper():
            time.sleep(0.01)

        self._set_robot_status_enabled(False)
        print("✓ Robot disabled successfully!")

    def get_control_mode(self) -> "PiperController.ControlMode":
        """Get the current control mode.

        Returns:
            Current control mode (PiperController.ControlMode.END_EFFECTOR or PiperController.ControlMode.JOINT_SPACE)
        """
        with self.state_lock:
            return self._control_mode

    def set_control_mode(self, mode: "PiperController.ControlMode") -> None:
        """Set the control mode.

        Args:
            mode: Control mode (PiperController.ControlMode.END_EFFECTOR or PiperController.ControlMode.JOINT_SPACE)
        """
        with self.state_lock:
            old_mode = self._control_mode
            self._control_mode = mode
            if self.debug_mode:
                print(f"Control mode changed: {old_mode.value} -> {mode.value}")

    def get_target_pose(self) -> list[float]:
        """Get the current target position.

        Returns:
            Current target position [x, y, z, rx, ry, rz] in mm and degrees
        """
        with self.position_lock:
            return self._target_pose.copy()

    def set_target_pose(
        self,
        position: list[float],
        orientation: list[float],
    ) -> None:
        """Set target target pose.

        Args:
            position: Target position [x, y, z] in millimeters
            orientation: Target orientation [rx, ry, rz] in degrees
        """
        with self.position_lock:
            self._target_pose[0] = position[0]
            self._target_pose[1] = position[1]
            self._target_pose[2] = position[2]

            self._target_pose[3] = orientation[0]
            self._target_pose[4] = orientation[1]
            self._target_pose[5] = orientation[2]

            if self.debug_mode:
                print(f"Target pose set: {self._target_pose}")

    def update_target_pose(
        self,
        linear_delta: list[float],
        angular_delta: list[float],
    ) -> None:
        """Update target position with relative deltas.

        Args:
            linear_delta: Change in position [dx, dy, dz] in millimeters
            angular_delta: Change in orientation [droll, dpitch, dyaw] in degrees
        """
        with self.position_lock:
            new_pose = self._target_pose.copy()

            # Update position (in mm)
            new_pose[0] += linear_delta[0]
            new_pose[1] += linear_delta[1]
            new_pose[2] += linear_delta[2]

            # Update orientation (in degrees)
            new_pose[3] += angular_delta[0]
            new_pose[4] += angular_delta[1]
            new_pose[5] += angular_delta[2]

            self._target_pose = new_pose

            if self.debug_mode:
                print(
                    f"Pose updated: pos=[{new_pose[0]:.1f}, {new_pose[1]:.1f}, {new_pose[2]:.1f}], "
                    f"rot=[{new_pose[3]:.1f}, {new_pose[4]:.1f}, {new_pose[5]:.1f}]"
                )

    def get_gripper_open_value(self) -> float:
        """Get the current gripper open value.

        Returns:
            Current gripper open value in degrees.
        """
        with self.position_lock:
            return self._gripper_open_value

    def set_gripper_open_value(self, gripper_open_value: float) -> None:
        """Update target gripper position.

        Args:
            gripper_open_value: Gripper open value in degrees.
        """
        with self.position_lock:
            # clamp gripper open value to limits
            new_gripper_open_value = max(
                self.GRIPPER_LIMITS[0], min(self.GRIPPER_LIMITS[1], gripper_open_value)
            )
            self._gripper_open_value = new_gripper_open_value

            if self.debug_mode:
                print(f"Gripper updated: {self._gripper_open_value}")

    def update_gripper_open_value(self, gripper_open_value_delta: float) -> None:
        """Update target gripper position.

        Args:
            gripper_open_value_delta: Gripper open value delta in degrees.
        """
        with self.position_lock:
            new_gripper_open_value = self._gripper_open_value + gripper_open_value_delta

            # clamp gripper open value to limits
            self._gripper_open_value = max(
                self.GRIPPER_LIMITS[0],
                min(self.GRIPPER_LIMITS[1], new_gripper_open_value),
            )

            if self.debug_mode:
                print(f"Gripper updated: {self._gripper_open_value}")

    def get_target_joint_angles(self) -> list[float]:
        """Get the current target joint angles.

        Returns:
            Current target joint angles [j1, j2, j3, j4, j5, j6] in degrees
        """
        with self.position_lock:
            return self._target_joint_angles.copy()

    def set_target_joint_angles(self, joint_angles: list[float]) -> None:
        """Set target joint angles.

        Args:
            joint_angles: Target joint angles [j1, j2, j3, j4, j5, j6] in degrees
        """
        with self.position_lock:
            # Clamp joint angles to limits
            clamped_angles = []
            for i, angle in enumerate(joint_angles):
                min_limit, max_limit = self.JOINT_LIMITS[i]
                clamped_angle = max(min_limit, min(max_limit, angle))
                clamped_angles.append(clamped_angle)

            self._target_joint_angles = clamped_angles

            if self.debug_mode:
                print(f"Target joint angles set: {self._target_joint_angles}")

    def update_target_joint_angles(self, joint_deltas: list[float]) -> None:
        """Update target joint angles with relative deltas.

        Args:
            joint_deltas: Change in joint angles [dj1, dj2, dj3, dj4, dj5, dj6] in degrees
        """
        with self.position_lock:
            new_joint_angles = []
            for i, delta in enumerate(joint_deltas):
                new_angle = self._target_joint_angles[i] + delta
                min_limit, max_limit = self.JOINT_LIMITS[i]
                clamped_angle = max(min_limit, min(max_limit, new_angle))
                new_joint_angles.append(clamped_angle)

            self._target_joint_angles = new_joint_angles

            if self.debug_mode:
                print(f"Joint angles updated: {self._target_joint_angles}")

    def move_to_home(self) -> bool:
        """Move robot to home position based on current control mode."""
        try:
            print("🏠 Moving to home position...")

            # Ensure robot is enabled
            enable_start_time = time.time()
            while not self.piper.EnablePiper():
                if time.time() - enable_start_time > 2:
                    print("Enable timeout. Attempting graceful stop...")
                    self.graceful_stop()
                    enable_start_time = time.time()
                time.sleep(0.01)

            current_mode = self.get_control_mode()
            with self.position_lock:
                if current_mode == PiperController.ControlMode.END_EFFECTOR:
                    self._target_pose = self.HOME_POSE.copy()
                    if self.debug_mode:
                        print("Moved to home pose (end effector mode)")
                elif current_mode == PiperController.ControlMode.JOINT_SPACE:
                    self._target_joint_angles = self.HOME_JOINT_ANGLES.copy()
                    if self.debug_mode:
                        print("Moved to home joint angles (joint space mode)")
                else:
                    print("Unknown control mode; cannot move to home position")
                    return False

            print("✓ Home position set")
            return True

        except Exception as e:
            print(f"✗ Home position error: {e}")
            return False

    def control_loop(self) -> None:
        """Main robot control loop running at robot_rate Hz.

        This should be run in a separate thread.
        Only sends commands when robot is in ENABLED state.
        """
        loop_period = 1.0 / self.robot_rate

        while self.running.is_set():
            try:
                # Only send commands if robot is enabled
                if self.is_robot_enabled():
                    # Get current control mode
                    current_mode = self.get_control_mode()

                    with self.position_lock:
                        if current_mode == PiperController.ControlMode.END_EFFECTOR:
                            # Send end-effector pose command
                            self._send_end_effector_command(self._target_pose)
                        elif current_mode == PiperController.ControlMode.JOINT_SPACE:
                            # Send joint angles command
                            self._send_joint_command(self._target_joint_angles)

                        # Always send gripper command regardless of control mode
                        self._send_gripper_command(self._gripper_open_value)
                else:
                    # Robot is not enabled, just sleep without sending commands
                    if self.debug_mode:
                        print("Control loop: Robot not enabled, skipping command")

                time.sleep(loop_period)

            except Exception as e:
                print(f"Robot control loop error: {e}")
                time.sleep(0.01)

    def _send_end_effector_command(self, command: list[float]) -> None:
        """Send end-effector pose command to the robot.

        Args:
            command: Command to the robot [x, y, z, rx, ry, rz, gripper]
        """
        try:
            # Convert from mm/degrees to piper SDK units (0.001mm/0.001degrees)
            X = round(command[0] * 1000)
            Y = round(command[1] * 1000)
            Z = round(command[2] * 1000)
            RX = round(command[3] * 1000)
            RY = round(command[4] * 1000)
            RZ = round(command[5] * 1000)

            # Set robot to position control mode (move_mode = 0x00)
            self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
            self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)

        except Exception as e:
            print(f"Failed to send end-effector command: {e}")

    def _send_joint_command(self, joint_angles: list[float]) -> None:
        """Send joint angles command to the robot.

        Args:
            joint_angles: Joint angles [j1, j2, j3, j4, j5, j6] in degrees
        """
        try:
            # Convert from degrees to piper SDK units (0.001degrees)
            joint_1 = round(joint_angles[0] * 1000)
            joint_2 = round(joint_angles[1] * 1000)
            joint_3 = round(joint_angles[2] * 1000)
            joint_4 = round(joint_angles[3] * 1000)
            joint_5 = round(joint_angles[4] * 1000)
            joint_6 = round(joint_angles[5] * 1000)

            # Set robot to joint control mode (move_mode = 0x01)
            self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
            self.piper.JointCtrl(joint_1, joint_2, joint_3, joint_4, joint_5, joint_6)

        except Exception as e:
            print(f"Failed to send joint command: {e}")

    def _send_gripper_command(self, gripper_open_value: float) -> None:
        """Send gripper command to the robot.

        Args:
            gripper_open_value: Gripper open value in degrees
        """
        try:
            # Convert from degrees to piper SDK units (0.001degrees)
            gripper_value = round(gripper_open_value * 1000)
            self.piper.GripperCtrl(gripper_value, 1000, 0x01, 0)

        except Exception as e:
            print(f"Failed to send gripper command: {e}")

    def emergency_stop(self) -> bool:
        """Emergency stop - immediately halt all motion."""
        try:
            print("🚨 EMERGENCY STOP!")

            # Send emergency stop commands to hardware
            self.piper.MotionCtrl_1(0x02, 0, 0)
            self.piper.GripperCtrl(0, 0, 0x00, 0)

            # Disable robot hardware
            self._disable_robot()

            print("✓ Emergency stop completed")
            return True

        except Exception as e:
            print(f"✗ Emergency stop error: {e}")
            return False

    def graceful_stop(self) -> bool:
        """Gracefully stop the robot and move to home position."""
        try:
            print("🛑 Graceful stop activated!")

            # Disable robot state (this will stop the control loop from sending commands)
            self._set_robot_status_enabled(False)

            time.sleep(0.1)

            # Motion control to go home
            self.piper.MotionCtrl_1(0x01, 0, 0)
            self.piper.GripperCtrl(0, 0, 0x00, 0)
            print(
                "Gracefully falling forward for 5 seconds. Please hold the robot if it doesn't reach home after this time!"
            )
            time.sleep(5)

            # Disable motion and gripper
            self.piper.MotionCtrl_1(0x02, 0, 0)

            print("Disabling robot...")
            self._disable_robot()

            print("✓ Graceful stop completed")
            return True

        except Exception as e:
            print(f"✗ Graceful stop error: {e}")
            return False

    def resume_robot(self) -> bool:
        """Resume robot operation by re-enabling it.

        This method can be called after emergency_stop() or graceful_stop()
        to re-enable the robot and resume normal operation.

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.is_robot_enabled():
                print("Robot is already enabled")
                return True

            print("🔄 Resuming robot...")

            # Re-enable the robot
            self._enable_robot()

            # After enabling, align controller targets to current measured state
            # to avoid any sudden jumps to stale targets
            try:
                current_mode = self.get_control_mode()
                if current_mode == PiperController.ControlMode.JOINT_SPACE:
                    joint_angles = self.get_current_joint_angles()
                    if joint_angles is not None and len(joint_angles) >= 6:
                        self.set_target_joint_angles(joint_angles)
                elif current_mode == PiperController.ControlMode.END_EFFECTOR:
                    end_pose = self.get_current_end_pose()
                    if end_pose is not None and len(end_pose) >= 6:
                        position = end_pose[:3]
                        orientation = end_pose[3:6]
                        self.set_target_pose(position, orientation)

                # Sync gripper if reading is available
                gripper_value = self.get_current_gripper_open_value()
                if gripper_value is not None:
                    self.set_gripper_open_value(gripper_value)
            except Exception as sync_err:
                if self.debug_mode:
                    print(f"Warning: failed to sync targets on resume: {sync_err}")

            print("✓ Robot resumed successfully!")
            return True

        except Exception as e:
            print(f"✗ Resume robot error: {e}")
            return False

    def get_current_end_pose(self) -> list[float] | None:
        """Get the current measured end effector pose from the robot, if available.

        Returns:
            list or None: [x, y, z, rx, ry, rz] or None if not available
        """
        if hasattr(self, "piper") and self.piper is not None:
            try:
                end_pose_msg = self.piper.GetArmEndPoseMsgs()
                if end_pose_msg:
                    return [
                        end_pose_msg.end_pose.X_axis / 1000,
                        end_pose_msg.end_pose.Y_axis / 1000,
                        end_pose_msg.end_pose.Z_axis / 1000,
                        end_pose_msg.end_pose.RX_axis / 1000,
                        end_pose_msg.end_pose.RY_axis / 1000,
                        end_pose_msg.end_pose.RZ_axis / 1000,
                    ]
            except Exception as e:
                if self.debug_mode:
                    print(f"Failed to get current end pose: {e}")
        return None

    def get_current_joint_angles(self) -> list[float] | None:
        """Get the current measured joint angles from the robot, if available.

        Returns:
            list or None: [j1, j2, j3, j4, j5, j6] or None if not available
        """
        if hasattr(self, "piper") and self.piper is not None:
            try:
                joint_msg = self.piper.GetArmJointMsgs().joint_state
                if joint_msg:
                    return [
                        joint_msg.joint_1 / 1000,
                        joint_msg.joint_2 / 1000,
                        joint_msg.joint_3 / 1000,
                        joint_msg.joint_4 / 1000,
                        joint_msg.joint_5 / 1000,
                        joint_msg.joint_6 / 1000,
                    ]
            except Exception as e:
                if self.debug_mode:
                    print(f"Failed to get current joint angles: {e}")
        return None

    def get_current_gripper_open_value(self) -> float | None:
        """Get the current measured gripper open value from the robot, if available.

        Returns:
            float or None: Current gripper open value or None if not available
        """
        if hasattr(self, "piper") and self.piper is not None:
            try:
                gripper_msg = self.piper.GetArmGripperMsgs()
                if gripper_msg:
                    return gripper_msg.gripper_state.grippers_angle / 1000
            except Exception as e:
                if self.debug_mode:
                    print(f"Failed to get current gripper open value: {e}")
        return None

    def get_robot_status(self) -> dict[str, Any]:
        """Get comprehensive robot status information.

        Returns:
            dict: Dictionary containing robot status information
        """
        try:
            status = {
                "enabled": self.is_robot_enabled(),
                "control_mode": self.get_control_mode(),
                "target_pose": self.get_target_pose(),
                "target_joint_angles": self.get_target_joint_angles(),
                "gripper_open_value": self.get_gripper_open_value(),
                "current_end_pose": self.get_current_end_pose(),
                "current_joint_angles": self.get_current_joint_angles(),
                "current_gripper_open_value": self.get_current_gripper_open_value(),
            }
            return status

        except Exception as e:
            print(f"Failed to get robot status: {e}")
            return {
                "enabled": None,
                "control_mode": None,
                "target_pose": None,
                "target_joint_angles": None,
                "gripper_open_value": None,
                "current_end_pose": None,
                "current_joint_angles": None,
                "current_gripper_open_value": None,
            }
