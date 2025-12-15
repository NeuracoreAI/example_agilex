"""Policy state - policy prediction, policy action, policy action index."""

import threading
from enum import Enum
from typing import Any

import numpy as np


class PolicyState:
    """Policy state - policy prediction, policy action, policy action index."""

    class ExecutionMode(Enum):
        """Execution mode enumeration."""

        TARGETING_TIME = "targeting_time"
        TARGETING_POSE = "targeting_pose"

    def __init__(self) -> None:
        """Initialize PolicyState with default values."""
        self._prediction_horizon_sync_points: list[Any] = []
        self._prediction_horizon_lock = threading.Lock()
        self._execution_ratio: float = 1.0

        self._policy_rgb_image_input: np.ndarray | None = None
        self._policy_rgb_image_input_lock = threading.Lock()

        self._policy_state_input: np.ndarray | None = None
        self._policy_state_input_lock = threading.Lock()

        self._ghost_robot_playing: bool = False
        self._ghost_action_index: int = 0

        # Policy execution state
        self._policy_inputs_locked: bool = False
        self._locked_prediction_horizon_sync_points: list[Any] = []
        self._execution_action_index: int = 0
        self._execution_lock = threading.Lock()

        # Continuous play and execution mode
        self._continuous_play_active: bool = False
        self._execution_mode: PolicyState.ExecutionMode = (
            PolicyState.ExecutionMode.TARGETING_TIME
        )

    def get_prediction_horizon_length(self) -> int:
        """Get prediction horizon length (thread-safe)."""
        with self._prediction_horizon_lock:
            return len(self._prediction_horizon_sync_points)

    def get_prediction_horizon_sync_points(self) -> list[Any]:
        """Get prediction horizon sync points (thread-safe)."""
        with self._prediction_horizon_lock:
            return list(self._prediction_horizon_sync_points)

    def set_prediction_horizon_sync_points(self, sync_points: list[Any]) -> None:
        """Set prediction horizon sync points (thread-safe)."""
        with self._prediction_horizon_lock:
            self._prediction_horizon_sync_points = list(sync_points)

    def set_execution_ratio(self, ratio: float) -> None:
        """Set execution ratio used when locking prediction horizon."""
        # Clamp to (0, 1] to avoid zero-length horizons
        clamped_ratio = float(np.clip(ratio, 1e-6, 1.0))
        with self._prediction_horizon_lock:
            self._execution_ratio = clamped_ratio

    def get_execution_ratio(self) -> float:
        """Get execution ratio (thread-safe)."""
        with self._prediction_horizon_lock:
            return self._execution_ratio

    def get_policy_rgb_image_input(self) -> np.ndarray | None:
        """Get policy RGB image (thread-safe)."""
        with self._policy_rgb_image_input_lock:
            return (
                self._policy_rgb_image_input.copy()
                if self._policy_rgb_image_input is not None
                else None
            )

    def set_policy_rgb_image_input(self, image: np.ndarray) -> None:
        """Set policy RGB image (thread-safe).

        Raises:
            RuntimeError: If policy inputs are locked (during execution).
        """
        with self._execution_lock:
            if self._policy_inputs_locked:
                raise RuntimeError("Policy inputs are locked during execution")
        with self._policy_rgb_image_input_lock:
            self._policy_rgb_image_input = image.copy() if image is not None else None

    def get_policy_state_input(self) -> np.ndarray | None:
        """Get policy state input (thread-safe)."""
        with self._policy_state_input_lock:
            return (
                self._policy_state_input.copy()
                if self._policy_state_input is not None
                else None
            )

    def set_policy_state_input(self, input: np.ndarray) -> None:
        """Set policy state input (thread-safe).

        Raises:
            RuntimeError: If policy inputs are locked (during execution).
        """
        with self._execution_lock:
            if self._policy_inputs_locked:
                raise RuntimeError("Policy inputs are locked during execution")
        with self._policy_state_input_lock:
            self._policy_state_input = input.copy() if input is not None else None

    def get_ghost_robot_playing(self) -> bool:
        """Get ghost robot playing (thread-safe)."""
        return self._ghost_robot_playing

    def set_ghost_robot_playing(self, playing: bool) -> None:
        """Set ghost robot playing (thread-safe)."""
        self._ghost_robot_playing = playing

    def get_ghost_action_index(self) -> int:
        """Get ghost action index (thread-safe)."""
        return self._ghost_action_index

    def set_ghost_action_index(self, index: int) -> None:
        """Set ghost action index (thread-safe)."""
        self._ghost_action_index = index

    def reset_ghost_action_index(self) -> None:
        """Reset ghost action index (thread-safe)."""
        self._ghost_action_index = 0

    # Policy execution methods
    def start_policy_execution(self) -> None:
        """Start policy execution by locking inputs and storing horizon (thread-safe)."""
        with self._prediction_horizon_lock:
            source_sync_points = list(self._prediction_horizon_sync_points)
            total = len(source_sync_points)
            if total == 0:
                locked_sync_points = []
            else:
                num_actions = int(total * self._execution_ratio)
                num_actions = max(1, min(num_actions, total))
                locked_sync_points = source_sync_points[:num_actions]
        with self._execution_lock:
            self._policy_inputs_locked = True
            self._execution_action_index = 0
            self._locked_prediction_horizon_sync_points = locked_sync_points

    def end_policy_execution(self) -> None:
        """Stop policy execution and unlock inputs (thread-safe)."""
        with self._execution_lock:
            self._policy_inputs_locked = False
            self._locked_prediction_horizon_sync_points = []
            self._execution_action_index = 0

    def get_locked_prediction_horizon_sync_points(self) -> list[Any]:
        """Get locked prediction horizon sync points (thread-safe)."""
        with self._execution_lock:
            return list(self._locked_prediction_horizon_sync_points)

    def get_locked_prediction_horizon_length(self) -> int:
        """Get locked prediction horizon length (thread-safe)."""
        with self._execution_lock:
            return len(self._locked_prediction_horizon_sync_points)

    def get_execution_action_index(self) -> int:
        """Get current execution action index (thread-safe)."""
        with self._execution_lock:
            return self._execution_action_index

    def increment_execution_action_index(self) -> None:
        """Increment execution action index (thread-safe)."""
        with self._execution_lock:
            self._execution_action_index += 1

    def get_continuous_play_active(self) -> bool:
        """Get continuous play active state (thread-safe)."""
        return self._continuous_play_active

    def set_continuous_play_active(self, active: bool) -> None:
        """Set continuous play active state (thread-safe)."""
        self._continuous_play_active = active

    def get_execution_mode(self) -> ExecutionMode:
        """Get execution mode (thread-safe)."""
        return self._execution_mode

    def set_execution_mode(self, mode: ExecutionMode) -> None:
        """Set execution mode (thread-safe)."""
        self._execution_mode = mode
