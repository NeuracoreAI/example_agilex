#!/usr/bin/env python3
"""LeRobot S0100 leader arm: connect, read, and map to a configured follower.

The class holds follower-specific config (limits, offsets, directions, joint
mapping) and can output mapped joint angles + gripper for any follower DOF.
Raw actions are still available via read() for debugging.
"""

import numpy as np
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig

# Fixed S0100 leader arm parameters (do not change per follower).
JOINT_ACTION_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
]
GRIPPER_ACTION_KEY = "gripper.pos"
NUM_JOINTS = 5
USE_DEGREES = True


class LerobotSO100LeaderArm:
    """LeRobot S0100 leader arm: read raw or mapped to a configured follower."""

    def __init__(self, port: str, calibration_id: str) -> None:
        """Configure the leader arm (does not connect).

        Args:
            port: Serial port (e.g. /dev/ttyACM0).
            calibration_id: Id used when calibrating (lerobot-calibrate --teleop.id=...).
        """
        self._config = SO100LeaderConfig(
            port=port,
            id=calibration_id,
            use_degrees=USE_DEGREES,
        )
        self._leader = SO100Leader(self._config)
        self._follower_limits_deg: np.ndarray | None = None
        self._follower_offsets_deg: np.ndarray | None = None
        self._follower_directions: np.ndarray | None = None
        self._leader_to_follower_joint: dict[int, int] | None = None
        self._fixed_joints: dict[int, float] = {}

    def connect(self, calibrate: bool = False) -> None:
        """Connect to the leader arm. Raises if port or calibration fails."""
        self._leader.connect(calibrate=calibrate)

    def disconnect(self) -> None:
        """Disconnect from the leader arm."""
        self._leader.disconnect()

    def read(self) -> dict[str, float]:
        """Read current joint angles (degrees) and gripper (0–100). Raw leader output."""
        return self._leader.get_action()

    def configure_follower(
        self,
        follower_limits_deg: np.ndarray,
        follower_offsets_deg: np.ndarray,
        follower_directions: np.ndarray,
        leader_to_follower_joint: dict[int, int] | list[int],
        fixed_joints: dict[int, float] | None = None,
    ) -> None:
        """Set follower mapping so read_mapped() returns follower-space angles.

        Args:
            follower_limits_deg: (n_follower_joints, 2) min/max in degrees per joint.
            follower_offsets_deg: (n_follower_joints,) offset per follower joint (leader 0° -> follower offset°).
            follower_directions: (n_follower_joints,) sign per follower joint (+1 or -1).
            leader_to_follower_joint: leader joint index -> follower joint index (dict or list of length NUM_JOINTS).
            fixed_joints: optional dict {follower_joint_index: value} for joints with no leader (e.g. {3: 0.0}).
        """
        n = follower_limits_deg.shape[0]
        assert follower_limits_deg.shape == (n, 2)
        assert follower_offsets_deg.shape == (n,)
        assert follower_directions.shape == (n,)
        if isinstance(leader_to_follower_joint, dict):
            assert set(leader_to_follower_joint.keys()) == set(range(NUM_JOINTS))
        else:
            assert len(leader_to_follower_joint) == NUM_JOINTS
        self._follower_limits_deg = np.asarray(follower_limits_deg, dtype=np.float64)
        self._follower_offsets_deg = np.asarray(follower_offsets_deg, dtype=np.float64)
        self._follower_directions = np.asarray(follower_directions, dtype=np.float64)
        self._leader_to_follower_joint = (
            dict(leader_to_follower_joint)
            if isinstance(leader_to_follower_joint, dict)
            else {i: v for i, v in enumerate(leader_to_follower_joint)}
        )
        self._fixed_joints = dict(fixed_joints) if fixed_joints else {}

    def read_mapped(self) -> tuple[np.ndarray, float]:
        """Read leader and return follower-space joint angles (degrees) and gripper open (0–1).

        Must call configure_follower() first. Clips to follower limits; fixed joints set as configured.
        """
        if (
            self._follower_limits_deg is None
            or self._follower_offsets_deg is None
            or self._follower_directions is None
            or self._leader_to_follower_joint is None
        ):
            raise RuntimeError(
                "configure_follower() must be called before read_mapped()"
            )
        raw = self.read()
        n = self._follower_limits_deg.shape[0]
        angles = np.zeros(n, dtype=np.float64)
        for fj, val in self._fixed_joints.items():
            angles[fj] = val
        for i, key in enumerate(JOINT_ACTION_KEYS):
            leader_val = raw.get(key, 0.0)
            fj = self._leader_to_follower_joint[i]
            lo, hi = self._follower_limits_deg[fj, 0], self._follower_limits_deg[fj, 1]
            angles[fj] = np.clip(
                leader_val * self._follower_directions[fj]
                + self._follower_offsets_deg[fj],
                lo,
                hi,
            )
        gripper = float(np.clip(raw.get(GRIPPER_ACTION_KEY, 50.0) / 100.0, 0.0, 1.0))
        return angles, gripper

    @property
    def is_connected(self) -> bool:
        """True if the leader arm is connected."""
        return self._leader.is_connected

    @staticmethod
    def joint_keys() -> list[str]:
        """Ordered list of body joint keys (no gripper)."""
        return list(JOINT_ACTION_KEYS)

    @staticmethod
    def gripper_key() -> str:
        """Key for gripper value in the action dict."""
        return GRIPPER_ACTION_KEY
