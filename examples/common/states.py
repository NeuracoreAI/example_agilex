"""State containers used by the Piper example suite."""

import threading
from enum import Enum

import numpy as np
from common.one_euro_filter import OneEuroFilterTransform


class RobotActivityState(Enum):
    """Robot activity state enumeration."""

    ENABLED = "ENABLED"
    HOMING = "HOMING"
    DISABLED = "DISABLED"
    POLICY_CONTROLLED = "POLICY_CONTROLLED"


class ControllerState:
    """Controller input state - Quest Reader writes, IK/Joint reads."""

    def __init__(self) -> None:
        """Initialize controller input state and filtering state."""
        self._lock = threading.Lock()
        self.min_cutoff: float = 1.0
        self.beta: float = 0.0
        self.d_cutoff: float = 1.0
        self.transform_raw: np.ndarray | None = None
        self.transform: np.ndarray | None = None
        self.grip_value: float = 0.0
        self.trigger_value: float = 0.0
        self._filter: OneEuroFilterTransform | None = None


class TeleopState:
    """Teleop activation state - manages teleop start/stop."""

    def __init__(self) -> None:
        """Initialize teleop activation and scaling state."""
        self._lock = threading.Lock()
        self.active: bool = False
        self.controller_initial_transform: np.ndarray | None = None
        self.robot_initial_transform: np.ndarray | None = None
        self.translation_scale: float = 1.0
        self.rotation_scale: float = 1.0
        self.slow_scaling_mode_enabled: bool = False


class RobotState:
    """Current robot state - joint angles, end effector pose, activity state."""

    def __init__(self) -> None:
        """Initialize robot telemetry and activity state."""
        self._lock = threading.Lock()
        self.joint_angles: np.ndarray | None = None
        self.joint_torques: np.ndarray | None = None
        self.end_effector_pose: np.ndarray | None = None
        self.current_gripper_open_value: float | None = None
        self.target_gripper_open_value: float | None = None
        self.activity_state: RobotActivityState = RobotActivityState.DISABLED


class IKState:
    """IK solution state - target joint angles, pose, metrics."""

    def __init__(self) -> None:
        """Initialize IK target and status information."""
        self._lock = threading.Lock()
        self.target_joint_angles: np.ndarray | None = None
        self.target_pose: np.ndarray | None = None
        self.solve_time_ms: float = 0.0
        self.success: bool = True


class CameraState:
    """Camera state - RGB images for one or more cameras."""

    def __init__(self) -> None:
        """Initialize camera image storage."""
        self._lock = threading.Lock()
        self.rgb_images: dict[str, np.ndarray] = {}
