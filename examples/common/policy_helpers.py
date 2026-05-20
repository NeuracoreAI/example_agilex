"""Shared helpers for AgileX Neuracore policy rollout examples."""

from __future__ import annotations

from typing import Any

import neuracore as nc
import numpy as np
from neuracore_types import (
    BatchedJointData,
    BatchedNCData,
    BatchedParallelGripperOpenAmountData,
    DataType,
    EmbodimentDescription,
)

from .configs import GRIPPER_NAME, JOINT_NAMES

DEFAULT_ROBOT_NAME = "AgileX PiPER"


def embodiment_names_ordered(spec: list[str] | dict[int, str]) -> list[str]:
    """Ordered channel names from an embodiment entry (list or index→name map)."""
    if isinstance(spec, dict):
        return [spec[i] for i in sorted(spec)]
    return list(spec)


def get_policy_embodiments(
    policy: Any,
) -> tuple[EmbodimentDescription, EmbodimentDescription | None]:
    """Read input/output embodiment specs resolved on the loaded policy."""
    if hasattr(policy, "_policy"):
        inner = policy._policy
        return inner.input_embodiment_description, inner.output_embodiment_description
    input_emb = getattr(policy, "input_embodiment_description", None)
    if input_emb is None:
        raise AttributeError(
            "Could not read input_embodiment_description from policy; "
            "use nc.policy(..., robot_name=...) without overriding embodiments."
        )
    output_emb = getattr(policy, "output_embodiment_description", None)
    return input_emb, output_emb


def print_policy_embodiments(
    input_embodiment: EmbodimentDescription,
    output_embodiment: EmbodimentDescription | None,
) -> None:
    """Print resolved policy embodiment channels."""
    print("\n📋 Policy input embodiment (from model):")
    for data_type, spec in input_embodiment.items():
        print(f"  {data_type.name}: {embodiment_names_ordered(spec)}")
    if output_embodiment is not None:
        print("\n📋 Policy output embodiment (from model):")
        for data_type, spec in output_embodiment.items():
            print(f"  {data_type.name}: {embodiment_names_ordered(spec)}")


def log_robot_state_for_policy(
    data_manager: Any,
    input_embodiment_description: EmbodimentDescription,
) -> bool:
    """Log only sensor streams the policy expects. Returns True if anything was logged."""
    logged_any = False

    if DataType.JOINT_POSITIONS in input_embodiment_description:
        current_joint_angles = data_manager.get_current_joint_angles()
        if current_joint_angles is not None:
            joint_angles_rad = np.radians(current_joint_angles)
            positions_by_name = {
                jn: float(ang) for jn, ang in zip(JOINT_NAMES, joint_angles_rad)
            }
            policy_joint_order = embodiment_names_ordered(
                input_embodiment_description[DataType.JOINT_POSITIONS]
            )
            nc.log_joint_positions(
                {jn: positions_by_name[jn] for jn in policy_joint_order}
            )
            logged_any = True

    if DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in input_embodiment_description:
        gripper_open_value = data_manager.get_current_gripper_open_value()
        if gripper_open_value is not None:
            for gripper_name in embodiment_names_ordered(
                input_embodiment_description[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS]
            ):
                nc.log_parallel_gripper_open_amount(gripper_name, gripper_open_value)
            logged_any = True

    if DataType.RGB_IMAGES in input_embodiment_description:
        for camera_name in embodiment_names_ordered(
            input_embodiment_description[DataType.RGB_IMAGES]
        ):
            rgb_image = data_manager.get_rgb_image(camera_name)
            if rgb_image is not None:
                nc.log_rgb(camera_name, rgb_image)
                logged_any = True

    return logged_any


def log_sync_step_for_policy(
    step: Any,
    input_embodiment_description: EmbodimentDescription,
) -> bool:
    """Log a synchronized dataset step using only channels the policy expects."""
    logged_any = False

    if DataType.JOINT_POSITIONS in input_embodiment_description:
        joint_data = step.data.get(DataType.JOINT_POSITIONS, {})
        joint_positions_dict = {
            joint_name: joint_data[joint_name].value
            for joint_name in embodiment_names_ordered(
                input_embodiment_description[DataType.JOINT_POSITIONS]
            )
            if joint_name in joint_data
        }
        if joint_positions_dict:
            nc.log_joint_positions(joint_positions_dict)
            logged_any = True

    if DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in input_embodiment_description:
        gripper_data = step.data.get(DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS, {})
        for gripper_name in embodiment_names_ordered(
            input_embodiment_description[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS]
        ):
            if gripper_name in gripper_data:
                nc.log_parallel_gripper_open_amount(
                    gripper_name, gripper_data[gripper_name].open_amount
                )
                logged_any = True

    if DataType.RGB_IMAGES in input_embodiment_description:
        rgb_data = step.data.get(DataType.RGB_IMAGES, {})
        for camera_name in embodiment_names_ordered(
            input_embodiment_description[DataType.RGB_IMAGES]
        ):
            if camera_name in rgb_data:
                nc.log_rgb(camera_name, np.array(rgb_data[camera_name].frame))
                logged_any = True

    return logged_any


def convert_predictions_to_horizon(
    predictions: dict[DataType, dict[str, BatchedNCData]],
) -> dict[str, list[float]]:
    """Convert policy predictions to a per-channel horizon dict (model-driven keys)."""
    horizon: dict[str, list[float]] = {}

    if DataType.JOINT_TARGET_POSITIONS in predictions:
        for joint_name, batched in predictions[DataType.JOINT_TARGET_POSITIONS].items():
            if isinstance(batched, BatchedJointData):
                horizon[joint_name] = batched.value[0, :, 0].cpu().numpy().tolist()

    if DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS in predictions:
        for gripper_name, batched in predictions[
            DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS
        ].items():
            if isinstance(batched, BatchedParallelGripperOpenAmountData):
                horizon[gripper_name] = (
                    batched.open_amount[0, :, 0].cpu().numpy().tolist()
                )

    return horizon


def horizon_length(horizon: dict[str, list[float]]) -> int:
    """Number of steps in a prediction horizon dict."""
    if not horizon:
        return 0
    return len(next(iter(horizon.values())))


def arm_joint_names_in_horizon(horizon: dict[str, list[float]]) -> list[str]:
    """Body joint names present in a horizon (excludes gripper channels)."""
    return [name for name in JOINT_NAMES if name in horizon]


def joint_targets_deg_at_index(
    horizon: dict[str, list[float]], index: int
) -> np.ndarray | None:
    """Arm joint targets in degrees at horizon index (Piper JOINT_NAMES order)."""
    if not all(jn in horizon for jn in JOINT_NAMES):
        return None
    if any(index >= len(horizon[jn]) for jn in JOINT_NAMES):
        return None
    rad = np.array([horizon[jn][index] for jn in JOINT_NAMES], dtype=np.float64)
    return np.degrees(rad)


def gripper_open_at_index(
    horizon: dict[str, list[float]],
    index: int,
    gripper_names: list[str] | None = None,
) -> float | None:
    """Gripper open amount in [0, 1] from the first matching horizon channel."""
    names = gripper_names or [GRIPPER_NAME]
    for gripper_name in names:
        if gripper_name in horizon and index < len(horizon[gripper_name]):
            return float(np.clip(horizon[gripper_name][index], 0.0, 1.0))
    return None


def urdf_cfg_from_horizon(
    horizon: dict[str, list[float]], index: int
) -> np.ndarray | None:
    """Joint configuration in radians for Viser URDF (JOINT_NAMES order)."""
    if not all(jn in horizon for jn in JOINT_NAMES):
        return None
    if any(index >= len(horizon[jn]) for jn in JOINT_NAMES):
        return None
    return np.array([float(horizon[jn][index]) for jn in JOINT_NAMES], dtype=np.float64)
