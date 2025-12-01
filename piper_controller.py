#!/usr/bin/env python3
"""Core Piper Robot Controller."""

import threading
import time
from enum import Enum
from typing import Any

import numpy as np
from piper_sdk import C_PiperInterface_V2  # type: ignore[attr-defined]
from scipy.spatial.transform import Rotation


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
        control_mode: "PiperController.ControlMode" = ControlMode.JOINT_SPACE,
        neutral_joint_angles: np.ndarray | None = None,
        neutral_end_effector_pose: np.ndarray | None = None,
        debug_mode: bool = False,
    ) -> None:
        """Initialize the robot controller.

        Args:
            can_interface: CAN interface for robot communication (default: 'can0')
            robot_rate: Robot control loop rate in Hz (default: 100.0)
            control_mode: Initial control mode (END_EFFECTOR or JOINT_SPACE)
            neutral_joint_angles: Neutral joint angles [j1, j2, j3, j4, j5, j6] in degrees (default: None)
            neutral_end_effector_pose: Neutral end effector pose as 4x4 transformation matrix (default: None)
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
        if neutral_end_effector_pose is not None:
            if neutral_end_effector_pose.shape == (4, 4):
                self.HOME_POSE = neutral_end_effector_pose.copy().astype(np.float64)
            else:
                raise ValueError(
                    "neutral_end_effector_pose must be a 4x4 transformation matrix"
                )
        else:
            # Convert default 6D pose to 4x4 matrix
            default_6d_pose = np.array(
                [-3.123, -125.085, 382.251, -78.132, 84.303, -169.496], dtype=np.float64
            )
            self.HOME_POSE = self._pose_6d_to_4x4(default_6d_pose)

        if neutral_joint_angles is not None:
            self.HOME_JOINT_ANGLES = np.array(neutral_joint_angles, dtype=np.float64)
        else:
            self.HOME_JOINT_ANGLES = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
            )

        # Gripper range in degrees (for internal SDK communication)
        self.GRIPPER_DEGREES_MIN = 0.000
        self.GRIPPER_DEGREES_MAX = 95.00
        self.GRIPPER_DEGREES_RANGE = self.GRIPPER_DEGREES_MAX - self.GRIPPER_DEGREES_MIN

        # Home gripper value in normalized form (53.8 / 95.0)
        self.HOME_GRIPPER_OPEN_VALUE_DEGREES = 53.800

        # End-effector target pose
        self._target_pose = self.HOME_POSE.copy()

        # Joint space target angles
        self._target_joint_angles = self.HOME_JOINT_ANGLES.copy()

        # Gripper target open value
        self._gripper_open_value_degrees = self.HOME_GRIPPER_OPEN_VALUE_DEGREES

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

        self.piper.SetSDKGripperRangeParam(0.0, 0.1)

        self.JOINT_LIMITS = np.degrees(
            np.array(
                [
                    self.piper.GetSDKJointLimitParam("j1"),
                    self.piper.GetSDKJointLimitParam("j2"),
                    self.piper.GetSDKJointLimitParam("j3"),
                    self.piper.GetSDKJointLimitParam("j4"),
                    self.piper.GetSDKJointLimitParam("j5"),
                    self.piper.GetSDKJointLimitParam("j6"),
                ]
            )
        )

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

    @staticmethod
    def _pose_6d_to_4x4(pose_6d: np.ndarray) -> np.ndarray:
        """Convert 6D pose [x, y, z, rx, ry, rz] to 4x4 transformation matrix.

        Args:
            pose_6d: 6D pose [x, y, z in mm, rx, ry, rz in degrees]

        Returns:
            4x4 transformation matrix
        """
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = pose_6d[:3]  # Position in mm
        # Convert Euler angles (degrees) to rotation matrix
        rot = Rotation.from_euler("xyz", pose_6d[3:6], degrees=True)
        transform[:3, :3] = rot.as_matrix()
        return transform

    @staticmethod
    def _pose_4x4_to_6d(transform: np.ndarray) -> np.ndarray:
        """Convert 4x4 transformation matrix to 6D pose [x, y, z, rx, ry, rz].

        Args:
            transform: 4x4 transformation matrix

        Returns:
            6D pose [x, y, z in mm, rx, ry, rz in degrees]
        """
        pose_6d = np.zeros(6, dtype=np.float64)
        pose_6d[:3] = transform[:3, 3]  # Position in mm
        # Convert rotation matrix to Euler angles (degrees)
        rot = Rotation.from_matrix(transform[:3, :3])
        pose_6d[3:6] = rot.as_euler("xyz", degrees=True)
        return pose_6d

    def get_target_pose(self) -> np.ndarray:
        """Get the current target pose.

        Returns:
            4x4 transformation matrix representing the target pose
        """
        with self.position_lock:
            return self._target_pose.copy()

    def set_target_pose(
        self,
        transform: np.ndarray,
    ) -> None:
        """Set target pose from a 4x4 transformation matrix.

        Args:
            transform: 4x4 transformation matrix
        """
        with self.position_lock:
            if transform.shape != (4, 4):
                raise ValueError("transform must be a 4x4 transformation matrix")
            self._target_pose = transform.copy().astype(np.float64)

            if self.debug_mode:
                pos = self._target_pose[:3, 3]
                rot_euler = Rotation.from_matrix(self._target_pose[:3, :3]).as_euler(
                    "xyz", degrees=True
                )
                print(
                    f"Target pose set: pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}], rot=[{rot_euler[0]:.1f}, {rot_euler[1]:.1f}, {rot_euler[2]:.1f}]"
                )

    def update_target_pose(
        self,
        linear_delta: np.ndarray,
        angular_delta: np.ndarray,
    ) -> None:
        """Update target position with relative deltas.

        Args:
            linear_delta: Change in position [dx, dy, dz] in millimeters
            angular_delta: Change in orientation [droll, dpitch, dyaw] in degrees
        """
        with self.position_lock:
            lin_delta = np.array(linear_delta, dtype=np.float64)
            ang_delta = np.array(angular_delta, dtype=np.float64)

            # Update position (in mm)
            self._target_pose[:3, 3] += lin_delta

            # Update orientation: apply rotation delta to current rotation
            current_rot = Rotation.from_matrix(self._target_pose[:3, :3])
            delta_rot = Rotation.from_euler("xyz", ang_delta, degrees=True)
            new_rot = current_rot * delta_rot
            self._target_pose[:3, :3] = new_rot.as_matrix()

            if self.debug_mode:
                pos = self._target_pose[:3, 3]
                rot_euler = new_rot.as_euler("xyz", degrees=True)
                print(
                    f"Pose updated: pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}], "
                    f"rot=[{rot_euler[0]:.1f}, {rot_euler[1]:.1f}, {rot_euler[2]:.1f}]"
                )

    def get_gripper_open_value(self) -> float:
        """Get the current gripper open value.

        Returns:
            Current gripper open value normalized (0.0 to 1.0).
        """
        with self.position_lock:
            # Convert from degrees to normalized (0.0 to 1.0)
            normalized = (
                self._gripper_open_value_degrees - self.GRIPPER_DEGREES_MIN
            ) / self.GRIPPER_DEGREES_RANGE
            return float(np.clip(normalized, 0.0, 1.0))

    def set_gripper_open_value(self, gripper_open_value: float) -> None:
        """Update target gripper position.

        Args:
            gripper_open_value: Gripper open value normalized (0.0 to 1.0).
        """
        with self.position_lock:
            # Clamp normalized value to [0.0, 1.0]
            normalized = float(np.clip(gripper_open_value, 0.0, 1.0))
            # Convert to degrees for internal storage
            self._gripper_open_value_degrees = (
                normalized * self.GRIPPER_DEGREES_RANGE + self.GRIPPER_DEGREES_MIN
            )

            if self.debug_mode:
                print(f"Gripper updated: {normalized:.3f} (normalized)")

    def update_gripper_open_value(self, gripper_open_value_delta: float) -> None:
        """Update target gripper position.

        Args:
            gripper_open_value_delta: Gripper open value delta normalized (-1.0 to 1.0).
        """
        with self.position_lock:
            # Get current normalized value
            current_normalized = (
                self._gripper_open_value_degrees - self.GRIPPER_DEGREES_MIN
            ) / self.GRIPPER_DEGREES_RANGE
            # Add delta and clamp to [0.0, 1.0]
            new_normalized = float(
                np.clip(current_normalized + gripper_open_value_delta, 0.0, 1.0)
            )
            # Convert back to degrees for internal storage
            self._gripper_open_value_degrees = (
                new_normalized * self.GRIPPER_DEGREES_RANGE + self.GRIPPER_DEGREES_MIN
            )

            if self.debug_mode:
                print(f"Gripper updated: {new_normalized:.3f} (normalized)")

    def get_target_joint_angles(self) -> np.ndarray:
        """Get the current target joint angles.

        Returns:
            Current target joint angles [j1, j2, j3, j4, j5, j6] in degrees
        """
        with self.position_lock:
            return self._target_joint_angles.copy()

    def set_target_joint_angles(self, joint_angles: np.ndarray) -> None:
        """Set target joint angles.

        Args:
            joint_angles: Target joint angles [j1, j2, j3, j4, j5, j6] in degrees
        """
        with self.position_lock:
            angles = np.array(joint_angles, dtype=np.float64)

            # Clamp joint angles to limits using numpy
            clamped_angles = np.clip(
                angles, self.JOINT_LIMITS[:, 0], self.JOINT_LIMITS[:, 1]
            )

            self._target_joint_angles = clamped_angles

            if self.debug_mode:
                print(f"Target joint angles set: {self._target_joint_angles}")

    def update_target_joint_angles(self, joint_deltas: np.ndarray) -> None:
        """Update target joint angles with relative deltas.

        Args:
            joint_deltas: Change in joint angles [dj1, dj2, dj3, dj4, dj5, dj6] in degrees
        """
        with self.position_lock:
            deltas = np.array(joint_deltas, dtype=np.float64)
            new_joint_angles = self._target_joint_angles + deltas

            # Clamp joint angles to limits using numpy
            self._target_joint_angles = np.clip(
                new_joint_angles, self.JOINT_LIMITS[:, 0], self.JOINT_LIMITS[:, 1]
            )

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
                        self._send_gripper_command(self._gripper_open_value_degrees)
                else:
                    # Robot is not enabled, just sleep without sending commands
                    if self.debug_mode:
                        print("Control loop: Robot not enabled, skipping command")

                time.sleep(loop_period)

            except Exception as e:
                print(f"Robot control loop error: {e}")
                time.sleep(0.01)

    def _send_end_effector_command(self, transform: np.ndarray) -> None:
        """Send end-effector pose command to the robot.

        Args:
            transform: 4x4 transformation matrix
        """
        try:
            # Convert 4x4 matrix to 6D pose for SDK
            pose_6d = self._pose_4x4_to_6d(transform)

            # Convert from mm/degrees to piper SDK units (0.001mm/0.001degrees)
            X = round(float(pose_6d[0] * 1000))
            Y = round(float(pose_6d[1] * 1000))
            Z = round(float(pose_6d[2] * 1000))
            RX = round(float(pose_6d[3] * 1000))
            RY = round(float(pose_6d[4] * 1000))
            RZ = round(float(pose_6d[5] * 1000))

            # Set robot to position control mode (move_mode = 0x00)
            self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
            self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)

        except Exception as e:
            print(f"Failed to send end-effector command: {e}")

    def _send_joint_command(self, joint_angles: np.ndarray) -> None:
        """Send joint angles command to the robot.

        Args:
            joint_angles: Joint angles [j1, j2, j3, j4, j5, j6] in degrees
        """
        try:
            # Convert from degrees to piper SDK units (0.001degrees)
            joint_1 = round(float(joint_angles[0] * 1000))
            joint_2 = round(float(joint_angles[1] * 1000))
            joint_3 = round(float(joint_angles[2] * 1000))
            joint_4 = round(float(joint_angles[3] * 1000))
            joint_5 = round(float(joint_angles[4] * 1000))
            joint_6 = round(float(joint_angles[5] * 1000))

            # Set robot to joint control mode (move_mode = 0x01)
            self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
            self.piper.JointCtrl(joint_1, joint_2, joint_3, joint_4, joint_5, joint_6)

        except Exception as e:
            print(f"Failed to send joint command: {e}")

    def _send_gripper_command(self, gripper_open_value_degrees: float) -> None:
        """Send gripper command to the robot.

        Args:
            gripper_open_value_degrees: Gripper open value in degrees
        """
        try:
            # Convert from degrees to piper SDK units (0.001degrees)
            gripper_value = round(gripper_open_value_degrees * 1000)
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
                    end_pose = self.get_current_end_effector_pose()
                    if end_pose is not None and end_pose.shape == (4, 4):
                        self.set_target_pose(end_pose)

                # Sync gripper if reading is available
                gripper_normalized = self.get_current_gripper_open_value()
                if gripper_normalized is not None:
                    self.set_gripper_open_value(gripper_normalized)
            except Exception as sync_err:
                if self.debug_mode:
                    print(f"Warning: failed to sync targets on resume: {sync_err}")

            print("✓ Robot resumed successfully!")
            return True

        except Exception as e:
            print(f"✗ Resume robot error: {e}")
            return False

    def get_current_end_effector_pose(self) -> np.ndarray | None:
        """Get the current measured end effector pose from the robot, if available.

        Returns:
            4x4 transformation matrix or None if not available
        """
        if hasattr(self, "piper") and self.piper is not None:
            try:
                end_pose_msg = self.piper.GetArmEndPoseMsgs()
                if end_pose_msg:
                    # Get 6D pose from SDK
                    pose_6d = np.array(
                        [
                            end_pose_msg.end_pose.X_axis / 1000,
                            end_pose_msg.end_pose.Y_axis / 1000,
                            end_pose_msg.end_pose.Z_axis / 1000,
                            end_pose_msg.end_pose.RX_axis / 1000,
                            end_pose_msg.end_pose.RY_axis / 1000,
                            end_pose_msg.end_pose.RZ_axis / 1000,
                        ],
                        dtype=np.float64,
                    )
                    # Convert to 4x4 matrix
                    return self._pose_6d_to_4x4(pose_6d)
            except Exception as e:
                if self.debug_mode:
                    print(f"Failed to get current end pose: {e}")
        return None

    def get_current_joint_angles(self) -> np.ndarray | None:
        """Get the current measured joint angles from the robot, if available.

        Returns:
            numpy array or None: [j1, j2, j3, j4, j5, j6] or None if not available
        """
        if hasattr(self, "piper") and self.piper is not None:
            try:
                joint_msg = self.piper.GetArmJointMsgs().joint_state
                if joint_msg:
                    return np.array(
                        [
                            joint_msg.joint_1 / 1000,
                            joint_msg.joint_2 / 1000,
                            joint_msg.joint_3 / 1000,
                            joint_msg.joint_4 / 1000,
                            joint_msg.joint_5 / 1000,
                            joint_msg.joint_6 / 1000,
                        ],
                        dtype=np.float64,
                    )
            except Exception as e:
                if self.debug_mode:
                    print(f"Failed to get current joint angles: {e}")
        return None

    def get_current_gripper_open_value(self) -> float | None:
        """Get the current measured gripper open value from the robot, if available.

        Returns:
            float or None: Current gripper open value normalized (0.0 to 1.0) or None if not available
        """
        if hasattr(self, "piper") and self.piper is not None:
            try:
                gripper_msg = self.piper.GetArmGripperMsgs()
                if gripper_msg:
                    gripper_degrees = gripper_msg.gripper_state.grippers_angle / 1000
                    # Convert from degrees to normalized (0.0 to 1.0)
                    normalized = (
                        gripper_degrees - self.GRIPPER_DEGREES_MIN
                    ) / self.GRIPPER_DEGREES_RANGE
                    return float(np.clip(normalized, 0.0, 1.0))
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
                "current_end_pose": self.get_current_end_effector_pose(),
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
