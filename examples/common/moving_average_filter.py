"""Signal filters for smoothing policy output actions."""

from collections import deque

import numpy as np


class MovingAverageFilter:
    """Moving-average filter over a sliding window of vector samples.

    Maintains a fixed-length window of the most recent samples and returns
    their element-wise mean. Used to smooth the policy's per-step joint
    targets before they are sent to the robot, reducing jitter (including
    the discontinuities at prediction-horizon boundaries during continuous
    play).
    """

    def __init__(self, window_size: int) -> None:
        """Initialize the filter.

        Args:
            window_size: Number of recent samples to average. Must be >= 1.
                A value of 1 disables smoothing (output == input).
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self._window_size = window_size
        self._buffer: deque[np.ndarray] = deque(maxlen=window_size)

    def is_empty(self) -> bool:
        """Return True if no samples have been pushed since the last reset."""
        return len(self._buffer) == 0

    def reset(self) -> None:
        """Clear all buffered samples."""
        self._buffer.clear()

    def prime(self, value: np.ndarray) -> None:
        """Fill the window with copies of value to avoid an initial lag spike.

        Args:
            value: Sample to pre-fill the window with (e.g. the robot's current
                measured joint angles at the start of execution).
        """
        sample = np.asarray(value, dtype=float)
        self._buffer.clear()
        for _ in range(self._window_size):
            self._buffer.append(sample.copy())

    def update(self, value: np.ndarray) -> np.ndarray:
        """Push a new sample and return the filtered (averaged) output.

        Args:
            value: New sample vector.

        Returns:
            Element-wise mean over the current window.
        """
        self._buffer.append(np.asarray(value, dtype=float))
        return np.mean(self._buffer, axis=0)
