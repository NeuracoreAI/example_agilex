"""1€ Filter (One Euro Filter) for adaptive low-pass filtering."""

import numpy as np
from scipy.spatial.transform import Rotation


class OneEuroFilter:
    """1€ Filter (One Euro Filter) for adaptive low-pass filtering.

    The 1€ Filter adapts its smoothing based on signal velocity:
    - Fast motion: Low smoothing (low latency, responsive)
    - Slow motion: High smoothing (high stability, reduces jitter)
    """

    def __init__(
        self,
        t0: float,
        x0: float,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ):
        """Initialize 1€ Filter.

        Args:
            t0: Initial timestamp (in seconds)
            x0: Initial value
            min_cutoff: Minimum cutoff frequency (stabilizes when holding still)
                       Higher = less lag, more jitter
            beta: Speed coefficient (reduces lag when moving)
                  Higher = less lag, but more jitter during motion
            d_cutoff: Cutoff frequency for derivative (speed) filtering
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = 0.0
        self.t_prev = t0

    def smoothing_factor(self, t_e: float, cutoff: float) -> float:
        """Calculate smoothing factor for exponential smoothing.

        Args:
            t_e: Time elapsed since last update (in seconds)
            cutoff: Cutoff frequency

        Returns:
            Smoothing factor (0.0-1.0)
        """
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a: float, x: float, x_prev: float) -> float:
        """Apply exponential smoothing.

        Args:
            a: Smoothing factor (0.0-1.0)
            x: Current value
            x_prev: Previous smoothed value

        Returns:
            Smoothed value
        """
        return a * x + (1 - a) * x_prev

    def __call__(self, timestamp: float, x: float) -> float:
        """Filter a new value.

        Args:
            t: Current timestamp (in seconds)
            x: Current noisy value

        Returns:
            Filtered value
        """
        t_e = timestamp - self.t_prev

        # Avoid division by zero in rare cases
        if t_e <= 0.0:
            return self.x_prev

        # Calculate the derivative (speed) of the signal
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(
            self.smoothing_factor(t_e, self.d_cutoff), dx, self.dx_prev
        )

        # Calculate the adaptive cutoff frequency
        # This is the magic part: Cutoff increases with speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Filter the signal using the adaptive cutoff
        x_hat = self.exponential_smoothing(
            self.smoothing_factor(t_e, cutoff), x, self.x_prev
        )

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = timestamp

        return x_hat

    def update_params(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        """Update filter parameters without resetting state.

        Args:
            min_cutoff: Minimum cutoff frequency
            beta: Speed coefficient
            d_cutoff: Cutoff frequency for derivative filtering
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff


class OneEuroFilterTransform:
    """1€ Filter for 4x4 transformation matrices.

    Applies separate 1€ Filters to position (3D) and orientation (quaternion).
    """

    def __init__(
        self,
        t0: float,
        transform: np.ndarray,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ):
        """Initialize 1€ Filter for transforms.

        Args:
            t0: Initial timestamp (in seconds)
            transform: Initial 4x4 transformation matrix
            min_cutoff: Minimum cutoff frequency
            beta: Speed coefficient
            d_cutoff: Cutoff frequency for derivative filtering
        """
        # Extract position and orientation
        position = transform[:3, 3]
        rotation = Rotation.from_matrix(transform[:3, :3])
        quat = rotation.as_quat()  # [x, y, z, w]

        # Create filters for position (3 components)
        self.position_filters = [
            OneEuroFilter(t0, pos, min_cutoff, beta, d_cutoff) for pos in position
        ]

        # Create filters for quaternion (4 components)
        self.quat_filters = [
            OneEuroFilter(t0, q, min_cutoff, beta, d_cutoff) for q in quat
        ]

        # Store previous quaternion for sign correction
        self.quat_prev = quat.copy()

        self.t_prev = t0

    def __call__(self, timestamp: float, transform: np.ndarray) -> np.ndarray:
        """Filter a new transform.

        Args:
            t: Current timestamp (in seconds)
            transform: Current 4x4 transformation matrix

        Returns:
            Filtered 4x4 transformation matrix
        """
        # Extract position and orientation
        position = transform[:3, 3]
        rotation = Rotation.from_matrix(transform[:3, :3])
        quat = rotation.as_quat()  # [x, y, z, w]

        # Fix quaternion sign to avoid sudden flips (q and -q represent same rotation)
        # Choose the quaternion closest to the previous one
        if np.dot(quat, self.quat_prev) < 0:
            quat = -quat

        # Filter position components
        filtered_position = np.array(
            [
                filter(timestamp, pos)
                for filter, pos in zip(self.position_filters, position)
            ]
        )

        # Filter quaternion components
        filtered_quat = np.array(
            [filter(timestamp, q) for filter, q in zip(self.quat_filters, quat)]
        )

        # Normalize quaternion (important!)
        filtered_quat = filtered_quat / np.linalg.norm(filtered_quat)

        # Convert back to rotation matrix
        filtered_rotation = Rotation.from_quat(filtered_quat)

        # Build filtered transform
        filtered_transform = np.eye(4)
        filtered_transform[:3, 3] = filtered_position
        filtered_transform[:3, :3] = filtered_rotation.as_matrix()

        self.quat_prev = filtered_quat.copy()
        self.t_prev = timestamp

        return filtered_transform

    def update_params(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        """Update filter parameters for all sub-filters without resetting state.

        Args:
            min_cutoff: Minimum cutoff frequency
            beta: Speed coefficient
            d_cutoff: Cutoff frequency for derivative filtering
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        for filter in self.position_filters + self.quat_filters:
            filter.update_params(min_cutoff, beta, d_cutoff)
