#!/usr/bin/env python3
"""Thread-safe teleoperation state management.

This module provides shared state classes for teleoperation systems that need
to coordinate between multiple threads (data collection, IK solving, visualization).
"""
import threading
import time
import queue  # Added for async queueing
from enum import Enum
from typing import Any, Callable

import numpy as np

from .configs import GRIPPER_NAME, JOINT_NAMES
from .one_euro_filter import OneEuroFilterTransform

import threading
import time
import queue
from typing import Any, Callable
import numpy as np

from .configs import GRIPPER_NAME, JOINT_NAMES
from .states import (
    RobotActivityState, ControllerState, TeleopState, 
    RobotState, IKState, CameraState
)

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
        # Teleoperation scaling parameters (how much controller motion maps to robot motion)
        self.translation_scale: float = 1.0
        self.rotation_scale: float = 1.0
        self.slow_scaling_mode_enabled: bool = False


class RobotState:
    """Current robot state - joint angles, end effector pose, activity state."""

    def __init__(self) -> None:
        """Initialize RobotState with default values."""
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
        """Initialize IKState with default values."""
        self._lock = threading.Lock()

        self.target_joint_angles: np.ndarray | None = None
        self.target_pose: np.ndarray | None = None
        self.solve_time_ms: float = 0.0
        self.success: bool = True


class CameraState:
    """Camera state - RGB images for one or more cameras."""

    def __init__(self) -> None:
        """Initialize CameraState with default values."""
        self._lock = threading.Lock()

        # Map from camera name -> latest RGB image
        self.rgb_images: dict[str, np.ndarray] = {}



class DataManager:
    """Main state container coordinating all state groups."""
    def __init__(self) -> None:
        self._controller_state = ControllerState()
        self._teleop_state = TeleopState()
        self._robot_state = RobotState()
        self._ik_state = IKState()
        self._camera_state = CameraState()

        self._shutdown_event = threading.Event()
        self._on_change_callback: Callable[[str, dict[str, Any], float], None] | None = None
        self._callback_queue: queue.Queue = queue.Queue(maxsize=60)
        
        self._worker_thread = threading.Thread(
            target=self._callback_worker_loop, 
            name="NeuracoreCallbackWorker", 
            daemon=True
        )
        self._worker_thread.start()
        
    # ... (Keep the rest of the getter/setter methods like get_rgb_image, etc.) ...

    def set_on_change_callback(
        self, on_change_callback: Callable[[str, dict[str, Any], float], None]
    ) -> None:
        """Set on change callback (thread-safe)."""
        self._on_change_callback = on_change_callback

    def _queue_callback(self, name: str, payload: dict[str, Any], timestamp: float) -> None:
        """Helper to push payloads into the execution queue without blocking."""
        if self._on_change_callback is None:
            return
            
        try:
            # put_nowait drops data into the memory queue instantly (0.0ms blocking)
            self._callback_queue.put_nowait((name, payload, timestamp))
        except queue.Full:
            # Prevents out-of-memory if disk halts completely, without freezing telemetry loops
            print(f"⚠️ Neuracore background queue full! Dropping log packet: {name}")

    def _callback_worker_loop(self) -> None:
        """Background thread worker loop dedicated solely to performing slow disk IO updates."""
        while not self._shutdown_event.is_set() or not self._callback_queue.empty():
            try:
                # Wait up to 100ms for a logging event
                name, payload, timestamp = self._callback_queue.get(timeout=0.1)
                
                if self._on_change_callback is not None:
                    # Execute Neuracore disk operation safely here on a separate core
                    self._on_change_callback(name, payload, timestamp)
                    
                self._callback_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error in background logging callback: {e}")

    # ============================================================================
    # Camera State Methods
    # ============================================================================

    def get_rgb_image(self, camera_name: str) -> np.ndarray | None:
        """Get RGB image for a specific camera (thread-safe)."""
        with self._camera_state._lock:
            if not self._camera_state.rgb_images:
                return None
            img = self._camera_state.rgb_images.get(camera_name)
            return img.copy() if img is not None else None

    def set_rgb_image(self, image: np.ndarray, camera_name: str) -> None:
        """Set RGB image for a specific camera (thread-safe and non-blocking)."""
        with self._camera_state._lock:
            self._camera_state.rgb_images[camera_name] = image.copy()
            
        if self._on_change_callback:
            img_copy = self._camera_state.rgb_images[camera_name].copy()
            # Queue it instead of executing directly! Camera loop returns immediately.
            self._queue_callback("log_rgb", {camera_name: img_copy}, time.time())

    # ============================================================================
    # Controller State Methods
    # ============================================================================

    def get_controller_data(self) -> tuple[np.ndarray | None, float, float]:
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
                self._controller_state.transform_raw = transform.copy()

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
                    self._controller_state._filter.update_params(
                        self._controller_state.min_cutoff,
                        self._controller_state.beta,
                        self._controller_state.d_cutoff,
                    )
                    self._controller_state.transform = self._controller_state._filter(
                        current_time, transform
                    )
            else:
                self._controller_state.transform = None
                self._controller_state.transform_raw = None
                self._controller_state._filter = None

    def set_controller_filter_params(
        self, min_cutoff: float, beta: float, d_cutoff: float
    ) -> None:
        with self._controller_state._lock:
            self._controller_state.min_cutoff = min_cutoff
            self._controller_state.beta = beta
            self._controller_state.d_cutoff = d_cutoff

    def get_controller_filter_params(self) -> tuple[float, float, float]:
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
        with self._teleop_state._lock:
            self._teleop_state.active = active
            self._teleop_state.controller_initial_transform = (
                controller_initial.copy() if controller_initial is not None else None
            )
            self._teleop_state.robot_initial_transform = (
                robot_initial.copy() if robot_initial is not None else None
            )

    def set_teleop_scaling(
        self, translation_scale: float, rotation_scale: float
    ) -> None:
        if translation_scale <= 0.0 or rotation_scale <= 0.0:
            return
        with self._teleop_state._lock:
            self._teleop_state.translation_scale = translation_scale
            self._teleop_state.rotation_scale = rotation_scale

    def get_teleop_scaling(self) -> tuple[float, float]:
        with self._teleop_state._lock:
            return (
                self._teleop_state.translation_scale,
                self._teleop_state.rotation_scale,
            )

    def get_teleop_active(self) -> bool:
        with self._teleop_state._lock:
            return self._teleop_state.active

    def set_slow_scaling_mode_enabled(self, enabled: bool) -> None:
        with self._teleop_state._lock:
            self._teleop_state.slow_scaling_mode_enabled = enabled

    def toggle_slow_scaling_mode_enabled(self) -> bool:
        with self._teleop_state._lock:
            self._teleop_state.slow_scaling_mode_enabled = (
                not self._teleop_state.slow_scaling_mode_enabled
            )
            return self._teleop_state.slow_scaling_mode_enabled

    def get_slow_scaling_mode_enabled(self) -> bool:
        with self._teleop_state._lock:
            return self._teleop_state.slow_scaling_mode_enabled

    def get_initial_robot_controller_transforms(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
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
        with self._robot_state._lock:
            return self._robot_state.activity_state

    def set_robot_activity_state(self, state: RobotActivityState) -> None:
        with self._robot_state._lock:
            self._robot_state.activity_state = state

    def get_current_joint_angles(self) -> np.ndarray | None:
        with self._robot_state._lock:
            return (
                self._robot_state.joint_angles.copy()
                if self._robot_state.joint_angles is not None
                else None
            )

    def set_current_joint_angles(self, angles: np.ndarray) -> None:
        with self._robot_state._lock:
            self._robot_state.joint_angles = angles.copy()
        if self._on_change_callback:
            angles = self._robot_state.joint_angles
            if angles is not None:
                payload = {
                    jn: float(np.radians(angles[i])) for i, jn in enumerate(JOINT_NAMES)
                }
                self._queue_callback("log_joint_positions", payload, time.time())

    def get_current_joint_torques(self) -> np.ndarray | None:
        with self._robot_state._lock:
            return (
                self._robot_state.joint_torques.copy()
                if self._robot_state.joint_torques is not None
                else None
            )

    def set_current_joint_torques(self, torques: np.ndarray) -> None:
        with self._robot_state._lock:
            self._robot_state.joint_torques = torques.copy()
        if self._on_change_callback:
            torques = self._robot_state.joint_torques
            if torques is not None:
                payload = {jn: float(torques[i]) for i, jn in enumerate(JOINT_NAMES)}
                self._queue_callback("log_joint_torques", payload, time.time())

    def get_current_end_effector_pose(self) -> np.ndarray | None:
        with self._robot_state._lock:
            return (
                self._robot_state.end_effector_pose.copy()
                if self._robot_state.end_effector_pose is not None
                else None
            )

    def set_current_end_effector_pose(self, pose: np.ndarray) -> None:
        with self._robot_state._lock:
            self._robot_state.end_effector_pose = pose.copy()

    def get_current_gripper_open_value(self) -> float | None:
        with self._robot_state._lock:
            return self._robot_state.current_gripper_open_value

    def set_current_gripper_open_value(self, value: float) -> None:
        with self._robot_state._lock:
            self._robot_state.current_gripper_open_value = value
        if self._on_change_callback:
            self._queue_callback(
                "log_parallel_gripper_open_amounts",
                {GRIPPER_NAME: value},
                time.time(),
            )

    def get_target_gripper_open_value(self) -> float | None:
        with self._robot_state._lock:
            return self._robot_state.target_gripper_open_value

    def set_target_gripper_open_value(self, value: float) -> None:
        with self._robot_state._lock:
            self._robot_state.target_gripper_open_value = value
        if self._on_change_callback:
            self._queue_callback(
                "log_parallel_gripper_target_open_amounts",
                {GRIPPER_NAME: self._robot_state.target_gripper_open_value},
                time.time(),
            )

    # ============================================================================
    # IK State Methods
    # ============================================================================

    def get_target_joint_angles(self) -> np.ndarray | None:
        with self._ik_state._lock:
            return (
                self._ik_state.target_joint_angles.copy()
                if self._ik_state.target_joint_angles is not None
                else None
            )

    def set_target_joint_angles(self, angles: np.ndarray) -> None:
        with self._ik_state._lock:
            self._ik_state.target_joint_angles = angles.copy()
        if self._on_change_callback:
            angles = self._ik_state.target_joint_angles
            if angles is not None:
                payload = {
                    jn: float(np.radians(angles[i])) for i, jn in enumerate(JOINT_NAMES)
                }
                self._queue_callback(
                    "log_joint_target_positions", payload, time.time()
                )

    def set_target_pose(self, transform: np.ndarray | None) -> None:
        with self._ik_state._lock:
            self._ik_state.target_pose = (
                transform.copy() if transform is not None else None
            )

    def get_target_pose(self) -> np.ndarray | None:
        with self._ik_state._lock:
            return (
                self._ik_state.target_pose.copy()
                if self._ik_state.target_pose is not None
                else None
            )

    def set_ik_solve_time_ms(self, time_ms: float) -> None:
        with self._ik_state._lock:
            self._ik_state.solve_time_ms = time_ms

    def set_ik_success(self, success: bool) -> None:
        with self._ik_state._lock:
            self._ik_state.success = success

    def get_ik_solve_time_ms(self) -> float:
        with self._ik_state._lock:
            return self._ik_state.solve_time_ms

    def get_ik_success(self) -> bool:
        with self._ik_state._lock:
            return self._ik_state.success

    # ============================================================================
    # System State Methods
    # ============================================================================

    def request_shutdown(self) -> None:
        """Request shutdown of all threads."""
        self._shutdown_event.set()

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown is requested."""
        return self._shutdown_event.is_set()