#!/usr/bin/env python3
"""Thread-safe teleoperation state management.

This module provides shared state classes for teleoperation systems that need
to coordinate between multiple threads (data collection, IK solving, visualization).
"""

import threading
import time
from enum import Enum
from typing import Callable

import numpy as np

from .one_euro_filter import OneEuroFilterTransform


class RobotActivityState(Enum):
    """Robot activity state enumeration."""

    ENABLED = "ENABLED"
    HOMING = "HOMING"
    DISABLED = "DISABLED"
    POLICY_CONTROLLED = "POLICY_CONTROLLED"


class ControllerState:
    """Controller input state - Quest Reader writes, IK/Joint reads."""

    def __init__(self) -> None:
        """Initialize ControllerState with default values."""
        self._lock = threading.Lock()

        # 1€ Filter parameters
        self.min_cutoff: float = 1.0  # Minimum cutoff frequency
        self.beta: float = 0.0  # Speed coefficient
        self.d_cutoff: float = 1.0  # Derivative cutoff frequency

        # Controller data
        self.transform_raw: np.ndarray | None = None
        self.transform: np.ndarray | None = None  # Smoothed transform
        self.grip_value: float = 0.0
        self.trigger_value: float = 0.0
        self._filter: OneEuroFilterTransform | None = None


class TeleopState:
    """Teleop activation state - manages teleop start/stop."""

    def __init__(self) -> None:
        """Initialize TeleopState with default values."""
        self._lock = threading.Lock()

        self.active: bool = False
        self.controller_initial_transform: np.ndarray | None = None
        self.robot_initial_transform: np.ndarray | None = None


class RobotState:
    """Current robot state - joint angles, end effector pose, activity state."""

    def __init__(self) -> None:
        """Initialize RobotState with default values."""
        self._lock = threading.Lock()

        self.joint_angles: np.ndarray | None = None
        self.end_effector_pose: np.ndarray | None = None
        self.current_gripper_open_value: float | None = None
        self.target_gripper_open_value: float | None = None
        self.activity_state: RobotActivityState = RobotActivityState.DISABLED


class IKState:
    """IK solution state - target joint angles, pose, metrics."""

    def __init__(self) -> None:
        """Initialize IKState with default values."""
        self._lock = threading.Lock()

        self.target_joint_angles: np.ndarray | None = None
        self.target_pose: np.ndarray | None = None
        self.solve_time_ms: float = 0.0
        self.success: bool = True


class CameraState:
    """Camera state - RGB image, depth image, point cloud."""

    def __init__(self) -> None:
        """Initialize CameraState with default values."""
        self._lock = threading.Lock()

        self.rgb_image: np.ndarray | None = None


class DataManager:
    """Main state container coordinating all state groups.

    This class manages shared data between threads:
    - Data collection thread: updates controller data
    - IK solver thread: reads controller data, updates joint solutions
    - Main thread: reads everything for visualization

    Uses separate locks for each state group to reduce contention.
    """

    def __init__(self) -> None:
        """Initialize DataManager with default values."""
        # State groups with individual locks
        self._controller_state = ControllerState()
        self._teleop_state = TeleopState()
        self._robot_state = RobotState()
        self._ik_state = IKState()
        self._camera_state = CameraState()

        # System state
        self._shutdown_event = threading.Event()

        # Callback for state changes (RGB, target joints, current joints)
        self._on_change_callback: Callable[[str, float, float], None] | None = None

    def set_on_change_callback(
        self, on_change_callback: Callable[[str, float, float], None]
    ) -> None:
        """Set on change callback (thread-safe)."""
        self._on_change_callback = on_change_callback

    # ============================================================================
    # Camera State Methods
    # ============================================================================

    def get_rgb_image(self) -> np.ndarray | None:
        """Get RGB image (thread-safe)."""
        with self._camera_state._lock:
            return (
                self._camera_state.rgb_image.copy()
                if self._camera_state.rgb_image is not None
                else None
            )

    def set_rgb_image(self, image: np.ndarray) -> None:
        """Set RGB image (thread-safe)."""
        with self._camera_state._lock:
            self._camera_state.rgb_image = image.copy()
        if self._on_change_callback:
            self._on_change_callback(
                "log_rgb", self._camera_state.rgb_image.copy(), time.time()
            )

    # ============================================================================
    # Controller State Methods
    # ============================================================================

    def get_controller_data(self) -> tuple[np.ndarray | None, float, float]:
        """Get current controller data (thread-safe).

        Returns:
            Tuple of (controller_transform, grip_value, trigger_value)
        """
        with self._controller_state._lock:
            return (
                (
                    self._controller_state.transform.copy()
                    if self._controller_state.transform is not None
                    else None
                ),
                self._controller_state.grip_value,
                self._controller_state.trigger_value,
            )

    def set_controller_data(
        self, transform: np.ndarray | None, grip: float, trigger: float
    ) -> None:
        """Set controller data (thread-safe).

        Args:
            transform: np.ndarray | None - 4x4 transformation matrix or None
            grip: float - grip value
            trigger: float - trigger value

        Raises:
            ValueError: If the transform is not a 4x4 matrix
            ValueError: If the grip value is not between 0.0 and 1.0
            ValueError: If the trigger value is not between 0.0 and 1.0
        """
        if transform is not None and transform.shape != (4, 4):
            raise ValueError("Transform must be a 4x4 matrix")
        if grip < 0.0 or grip > 1.0:
            raise ValueError("Grip value must be between 0.0 and 1.0")
        if trigger < 0.0 or trigger > 1.0:
            raise ValueError("Trigger value must be between 0.0 and 1.0")

        with self._controller_state._lock:
            self._controller_state.grip_value = grip
            self._controller_state.trigger_value = trigger

            if transform is not None:
                current_time = time.time()

                # Store raw transform
                self._controller_state.transform_raw = transform.copy()

                # Initialize filter if needed
                if self._controller_state._filter is None:
                    self._controller_state._filter = OneEuroFilterTransform(
                        current_time,
                        transform,
                        self._controller_state.min_cutoff,
                        self._controller_state.beta,
                        self._controller_state.d_cutoff,
                    )
                    self._controller_state.transform = transform.copy()
                else:
                    # Update filter parameters if they changed
                    self._controller_state._filter.update_params(
                        self._controller_state.min_cutoff,
                        self._controller_state.beta,
                        self._controller_state.d_cutoff,
                    )

                    # Apply filter
                    self._controller_state.transform = self._controller_state._filter(
                        current_time, transform
                    )
            else:
                self._controller_state.transform = None
                self._controller_state.transform_raw = None
                self._controller_state._filter = (
                    None  # Reset filter when transform is None
                )

    def set_controller_filter_params(
        self, min_cutoff: float, beta: float, d_cutoff: float
    ) -> None:
        """Update 1€ Filter parameters for controller transform (thread-safe).

        Args:
            min_cutoff: Minimum cutoff frequency (stabilizes when holding still)
            beta: Speed coefficient (reduces lag when moving)
            d_cutoff: Cutoff frequency for derivative filtering
        """
        with self._controller_state._lock:
            self._controller_state.min_cutoff = min_cutoff
            self._controller_state.beta = beta
            self._controller_state.d_cutoff = d_cutoff

    def get_controller_filter_params(self) -> tuple[float, float, float]:
        """Get 1€ Filter parameters for controller transform (thread-safe).

        Returns:
            Tuple of (min_cutoff, beta, d_cutoff)
        """
        with self._controller_state._lock:
            return (
                self._controller_state.min_cutoff,
                self._controller_state.beta,
                self._controller_state.d_cutoff,
            )

    # ============================================================================
    # Teleop State Methods
    # ============================================================================

    def set_teleop_state(
        self,
        active: bool,
        controller_initial: np.ndarray | None,
        robot_initial: np.ndarray | None,
    ) -> None:
        """Set teleoperation state (thread-safe).

        Args:
            active: bool - whether teleop is active
            controller_initial: np.ndarray | None - 4x4 transformation matrix for initial controller transform or None to clear
            robot_initial: np.ndarray | None - 4x4 transformation matrix for initial robot transform or None to clear
        """
        with self._teleop_state._lock:
            self._teleop_state.active = active
            self._teleop_state.controller_initial_transform = (
                controller_initial.copy() if controller_initial is not None else None
            )
            self._teleop_state.robot_initial_transform = (
                robot_initial.copy() if robot_initial is not None else None
            )

    def get_teleop_active(self) -> bool:
        """Get teleoperation active state (thread-safe)."""
        with self._teleop_state._lock:
            return self._teleop_state.active

    def get_initial_robot_controller_transforms(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Get initial robot and controller transforms.

        These two transforms are captured on rising edge of grip button
        and reset on falling edge of grip button. (thread-safe)

        Returns:
            Tuple of (controller_initial_transform, robot_initial_transform)
        """
        with self._teleop_state._lock:
            return (
                (
                    self._teleop_state.controller_initial_transform.copy()
                    if self._teleop_state.controller_initial_transform is not None
                    else None
                ),
                (
                    self._teleop_state.robot_initial_transform.copy()
                    if self._teleop_state.robot_initial_transform is not None
                    else None
                ),
            )

    # ============================================================================
    # Robot State Methods
    # ============================================================================

    def get_robot_activity_state(self) -> RobotActivityState:
        """Get robot activity state (thread-safe).

        Returns:
            RobotActivityState - current robot activity state
        """
        with self._robot_state._lock:
            return self._robot_state.activity_state

    def set_robot_activity_state(self, state: RobotActivityState) -> None:
        """Set robot activity state (thread-safe).

        Args:
            state: RobotActivityState - new robot activity state
        """
        with self._robot_state._lock:
            self._robot_state.activity_state = state

    def get_current_joint_angles(self) -> np.ndarray | None:
        """Get current joint angles (thread-safe).

        Returns:
            Current joint angles or None if not available
        """
        with self._robot_state._lock:
            return (
                self._robot_state.joint_angles.copy()
                if self._robot_state.joint_angles is not None
                else None
            )

    def set_current_joint_angles(self, angles: np.ndarray) -> None:
        """Set current joint angles (thread-safe).

        Args:
            angles: np.ndarray - current joint angles
        """
        with self._robot_state._lock:
            self._robot_state.joint_angles = angles.copy()
        if self._on_change_callback:
            self._on_change_callback(
                "log_joint_positions",
                self._robot_state.joint_angles.copy(),
                time.time(),
            )

    def get_current_end_effector_pose(self) -> np.ndarray | None:
        """Get current end effector pose (thread-safe).

        Returns:
            Current end effector pose or None if not available
        """
        with self._robot_state._lock:
            return (
                self._robot_state.end_effector_pose.copy()
                if self._robot_state.end_effector_pose is not None
                else None
            )

    def set_current_end_effector_pose(self, pose: np.ndarray) -> None:
        """Set current end effector pose (thread-safe).

        Args:
            pose: np.ndarray - current end effector pose
        """
        with self._robot_state._lock:
            self._robot_state.end_effector_pose = pose.copy()

    def get_current_gripper_open_value(self) -> float | None:
        """Get current gripper open value (thread-safe).

        Returns:
            Current gripper open value or None if not available
        """
        with self._robot_state._lock:
            return (
                self._robot_state.current_gripper_open_value
                if self._robot_state.current_gripper_open_value is not None
                else None
            )

    def set_current_gripper_open_value(self, value: float) -> None:
        """Set current gripper open value (thread-safe).

        Args:
            value: float - current gripper open value
        """
        with self._robot_state._lock:
            self._robot_state.current_gripper_open_value = value

    def get_target_gripper_open_value(self) -> float | None:
        """Get target gripper open value (thread-safe).

        Returns:
            Target gripper open value or None if not available
        """
        with self._robot_state._lock:
            return self._robot_state.target_gripper_open_value

    def set_target_gripper_open_value(self, value: float) -> None:
        """Set target gripper open value (thread-safe).

        Args:
            value: float - target gripper open value
        """
        with self._robot_state._lock:
            self._robot_state.target_gripper_open_value = value
        if self._on_change_callback:
            self._on_change_callback(
                "log_parallel_gripper_open_amounts",
                self._robot_state.target_gripper_open_value,
                time.time(),
            )

    # ============================================================================
    # IK State Methods
    # ============================================================================

    def get_target_joint_angles(self) -> np.ndarray | None:
        """Get current joint configuration (thread-safe).

        Returns:
            Current target joint angles or None if not available
        """
        with self._ik_state._lock:
            return (
                self._ik_state.target_joint_angles.copy()
                if self._ik_state.target_joint_angles is not None
                else None
            )

    def set_target_joint_angles(self, angles: np.ndarray) -> None:
        """Set target joint angles (thread-safe).

        Args:
            angles: np.ndarray - target joint angles
        """
        with self._ik_state._lock:
            self._ik_state.target_joint_angles = angles.copy()
        if self._on_change_callback:
            self._on_change_callback(
                "log_joint_target_positions",
                self._ik_state.target_joint_angles.copy(),
                time.time(),
            )

    def set_target_pose(self, transform: np.ndarray | None) -> None:
        """Set target transform for visualization (thread-safe).

        Args:
            transform: np.ndarray | None - 4x4 transformation matrix or None to clear target transform
        """
        with self._ik_state._lock:
            self._ik_state.target_pose = (
                transform.copy() if transform is not None else None
            )

    def get_target_pose(self) -> np.ndarray | None:
        """Get target transform for visualization (thread-safe).

        Returns:
            Target transform or None if target transform is not set
        """
        with self._ik_state._lock:
            return (
                self._ik_state.target_pose.copy()
                if self._ik_state.target_pose is not None
                else None
            )

    def set_ik_solve_time_ms(self, time_ms: float) -> None:
        """Set IK solve time (thread-safe).

        Args:
            time_ms: float - IK solve time in milliseconds
        """
        with self._ik_state._lock:
            self._ik_state.solve_time_ms = time_ms

    def set_ik_success(self, success: bool) -> None:
        """Set IK success (thread-safe).

        Args:
            success: bool - True if IK was successful, False otherwise
        """
        with self._ik_state._lock:
            self._ik_state.success = success

    def get_ik_solve_time_ms(self) -> float:
        """Get IK solve time (thread-safe).

        Returns:
            IK solve time in milliseconds
        """
        with self._ik_state._lock:
            return self._ik_state.solve_time_ms

    def get_ik_success(self) -> bool:
        """Get IK success (thread-safe).

        Returns:
            True if IK was successful, False otherwise
        """
        with self._ik_state._lock:
            return self._ik_state.success

    # ============================================================================
    # System State Methods
    # ============================================================================

    def request_shutdown(self) -> None:
        """Request shutdown of all threads (lock-free using Event)."""
        self._shutdown_event.set()

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown is requested (lock-free using Event).

        Returns:
            True if shutdown is requested, False otherwise
        """
        return self._shutdown_event.is_set()
