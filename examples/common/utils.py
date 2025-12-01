"""Utility functions for the examples."""

import numpy as np
from scipy.spatial.transform import Rotation


def scale_and_add_delta_transform(
    delta_position: np.ndarray,
    delta_orientation: np.ndarray,
    translation_scale: float,
    rotation_scale: float,
    initial_transform: np.ndarray,
) -> np.ndarray:
    """Scale and add delta position and orientation to an initial transform.

    Args:
        delta_position: 3D position vector
        delta_orientation: 3x3 rotation matrix
        translation_scale: translation scale
        rotation_scale: rotation scale
        initial_transform: 4x4 transformation matrix

    Returns:
        4x4 transformation matrix
    """
    scaled_delta_position = delta_position * translation_scale

    # Scale the delta rotation by converting to axis-angle, scaling the angle, then converting back
    delta_rotation = Rotation.from_matrix(delta_orientation)
    delta_axis_angle = delta_rotation.as_rotvec()
    scaled_delta_axis_angle = delta_axis_angle * rotation_scale
    scaled_delta_rotation = Rotation.from_rotvec(scaled_delta_axis_angle).as_matrix()

    # Compose rotations by matrix multiplication (correct way to combine rotations)
    initial_rotation = initial_transform[:3, :3]
    new_rotation = initial_rotation @ scaled_delta_rotation

    target_transform = np.eye(4)
    target_transform[:3, 3] = initial_transform[:3, 3] + scaled_delta_position
    target_transform[:3, :3] = new_rotation

    return target_transform
