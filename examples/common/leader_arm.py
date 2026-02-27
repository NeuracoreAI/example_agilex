#!/usr/bin/env python3
"""SO100 leader arm: connect, read, and map to a configured follower.

This module avoids runtime dependency on lerobot by talking to the motors
directly through scservo_sdk and loading calibration JSON from the same default
cache path used by lerobot tools.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scservo_sdk as scs

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

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
MOTOR_IDS = {name: idx + 1 for idx, name in enumerate(MOTOR_NAMES)}

MOTOR_MODEL_NUMBER = 777
MOTOR_RESOLUTION = 4096
BAUDRATE = 1_000_000
TIMEOUT_MS = 1000
PROTOCOL_VERSION = 0

SIGN_BIT = 15
ADDR_RETURN_DELAY_TIME = (7, 1)
ADDR_MAXIMUM_ACCELERATION = (85, 1)
ADDR_ACCELERATION = (41, 1)
ADDR_OPERATING_MODE = (33, 1)
ADDR_TORQUE_ENABLE = (40, 1)
ADDR_LOCK = (55, 1)
ADDR_PRESENT_POSITION = (56, 2)

_HF_HOME = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", _HF_HOME / "lerobot"))
_CALIBRATION_DIR = Path(
    os.getenv("HF_LEROBOT_CALIBRATION", _LEROBOT_HOME / "calibration")
)


@dataclass
class MotorCalibration:
    """Per-motor calibration loaded from JSON."""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


def _decode_sign_magnitude(encoded: int, sign_bit: int = SIGN_BIT) -> int:
    direction_bit = (encoded >> sign_bit) & 1
    magnitude = encoded & ((1 << sign_bit) - 1)
    return -magnitude if direction_bit else magnitude


def _raw_to_degrees(raw: int, range_min: int, range_max: int) -> float:
    mid = (range_min + range_max) / 2
    return (raw - mid) * 360.0 / (MOTOR_RESOLUTION - 1)


def _raw_to_percent(raw: int, range_min: int, range_max: int) -> float:
    bounded = min(range_max, max(range_min, raw))
    return ((bounded - range_min) / (range_max - range_min)) * 100.0


def _load_leader_calibration(
    calibration_id: str,
) -> dict[str, MotorCalibration] | None:
    cal_file = (
        _CALIBRATION_DIR / "teleoperators" / "so_leader" / f"{calibration_id}.json"
    )
    if not cal_file.is_file():
        return None

    with open(cal_file, encoding="utf-8") as file:
        raw: dict = json.load(file)

    calibration: dict[str, MotorCalibration] = {}
    for motor_name, values in raw.items():
        calibration[motor_name] = MotorCalibration(
            id=values["id"],
            drive_mode=values["drive_mode"],
            homing_offset=values["homing_offset"],
            range_min=values["range_min"],
            range_max=values["range_max"],
        )
    return calibration


def _to_bytes(value: int, length: int) -> list[int]:
    if length == 1:
        return [value & 0xFF]
    if length == 2:
        return [scs.SCS_LOBYTE(value), scs.SCS_HIBYTE(value)]
    raise ValueError(f"Unsupported byte length: {length}")


def _patched_set_packet_timeout(port_handler: Any, packet_length: int) -> None:
    """Replacement for PortHandler.setPacketTimeout with relaxed timing."""
    port_handler.packet_timeout = (
        port_handler.tx_time_per_byte * packet_length
        + port_handler.tx_time_per_byte * 3.0
        + 50
    )


class _FeetechLeaderBus:
    """Minimal motor-bus helper for SO100 leader arm reads."""

    def __init__(self, port: str, calibration: dict[str, MotorCalibration]) -> None:
        self.port = port
        self.calibration = calibration
        self.motor_ids = dict(MOTOR_IDS)
        self.port_handler = scs.PortHandler(port)
        self.port_handler.setPacketTimeout = _patched_set_packet_timeout.__get__(
            self.port_handler, type(self.port_handler)
        )
        self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION)
        self.sync_reader = scs.GroupSyncRead(
            self.port_handler, self.packet_handler, 0, 0
        )
        self._connected = False

    def connect(self) -> None:
        if not self.port_handler.openPort():
            raise ConnectionError(f"Failed to open leader port {self.port}")
        self.port_handler.setBaudRate(BAUDRATE)
        self.port_handler.setPacketTimeoutMillis(TIMEOUT_MS)
        self._verify_motors()
        self._configure_leader()
        self._connected = True

    def _verify_motors(self) -> None:
        missing = []
        for name, motor_id in self.motor_ids.items():
            model_number, comm, _ = self.packet_handler.ping(
                self.port_handler, motor_id
            )
            if comm != scs.COMM_SUCCESS:
                missing.append(f"  - ID {motor_id} ({name}): no response")
            elif model_number != MOTOR_MODEL_NUMBER:
                missing.append(
                    f"  - ID {motor_id} ({name}): wrong model {model_number}"
                )

        if missing:
            raise ConnectionError(
                f"Leader motor check failed on {self.port}:\n" + "\n".join(missing)
            )

    def _write_register(
        self, motor_name: str, addr: int, length: int, value: int, num_retry: int = 2
    ) -> None:
        motor_id = self.motor_ids[motor_name]
        data = _to_bytes(value, length)
        for _ in range(1 + num_retry):
            comm, _ = self.packet_handler.writeTxRx(
                self.port_handler, motor_id, addr, length, data
            )
            if comm == scs.COMM_SUCCESS:
                time.sleep(0.005)
                return
            time.sleep(0.02)
        msg = self.packet_handler.getTxRxResult(comm)
        raise ConnectionError(
            f"Failed register write on motor '{motor_name}' (ID {motor_id}): {msg}"
        )

    def _disable_torque(self) -> None:
        for name in self.motor_ids:
            self._write_register(name, *ADDR_TORQUE_ENABLE, 0)
            self._write_register(name, *ADDR_LOCK, 0)

    def _configure_leader(self) -> None:
        self._disable_torque()
        for name in self.motor_ids:
            self._write_register(name, *ADDR_RETURN_DELAY_TIME, 0)
            self._write_register(name, *ADDR_MAXIMUM_ACCELERATION, 254)
            self._write_register(name, *ADDR_ACCELERATION, 254)
            self._write_register(name, *ADDR_OPERATING_MODE, 0)

    def disconnect(self) -> None:
        try:
            self._disable_torque()
        except Exception:
            pass
        self.port_handler.closePort()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self.port_handler.is_open

    def read_positions(self, num_retry: int = 3) -> dict[str, float]:
        addr, length = ADDR_PRESENT_POSITION
        comm = None
        for _ in range(1 + num_retry):
            self.sync_reader.clearParam()
            self.sync_reader.start_address = addr
            self.sync_reader.data_length = length
            for motor_id in self.motor_ids.values():
                self.sync_reader.addParam(motor_id)

            comm = self.sync_reader.txRxPacket()
            if comm == scs.COMM_SUCCESS:
                break
            time.sleep(0.01)

        if comm != scs.COMM_SUCCESS:
            msg = self.packet_handler.getTxRxResult(comm)
            raise ConnectionError(f"Leader sync read failed on {self.port}: {msg}")

        result: dict[str, float] = {}
        for name, motor_id in self.motor_ids.items():
            raw = self.sync_reader.getData(motor_id, addr, length)
            decoded = _decode_sign_magnitude(raw)
            calibration = self.calibration[name]
            if name == "gripper":
                value = _raw_to_percent(
                    decoded, calibration.range_min, calibration.range_max
                )
            else:
                value = _raw_to_degrees(
                    decoded, calibration.range_min, calibration.range_max
                )
            result[name] = float(value)
        return result


class SO100LeaderArm:
    """SO100 leader arm: read raw or mapped to a configured follower."""

    def __init__(self, port: str, calibration_id: str) -> None:
        """Configure the leader arm (does not connect)."""
        self._port = port
        self._calibration_id = calibration_id
        self._bus: _FeetechLeaderBus | None = None
        self._follower_limits_deg: np.ndarray | None = None
        self._follower_offsets_deg: np.ndarray | None = None
        self._follower_directions: np.ndarray | None = None
        self._leader_to_follower_joint: dict[int, int] | None = None
        self._fixed_joints: dict[int, float] = {}

    def connect(self, calibrate: bool = False) -> None:
        """Connect to the leader arm. Raises if port or calibration fails."""
        if calibrate:
            raise RuntimeError(
                "calibrate=True is not supported by this script. "
                "Run calibration separately, then connect with calibrate=False."
            )

        calibration = _load_leader_calibration(self._calibration_id)
        if calibration is None:
            expected = _CALIBRATION_DIR / "teleoperators" / "so_leader"
            raise RuntimeError(
                f"no calibration registered for leader id '{self._calibration_id}'. "
                f"Expected JSON under: {expected}"
            )

        missing_motors = [name for name in MOTOR_NAMES if name not in calibration]
        if missing_motors:
            raise RuntimeError(
                "invalid calibration file: missing motors " + ", ".join(missing_motors)
            )

        self._bus = _FeetechLeaderBus(self._port, calibration)
        try:
            self._bus.connect()
        except Exception:
            self._bus.disconnect()
            self._bus = None
            raise

    def disconnect(self) -> None:
        """Disconnect from the leader arm."""
        if self._bus is not None:
            self._bus.disconnect()
            self._bus = None

    def read(self) -> dict[str, float]:
        """Read current joint angles (degrees) and gripper (0–100). Raw leader output."""
        if self._bus is None or not self._bus.is_connected:
            raise RuntimeError("Leader arm is not connected")

        positions = self._bus.read_positions()
        action = {
            key: float(positions.get(name, 0.0))
            for key, name in zip(JOINT_ACTION_KEYS, MOTOR_NAMES[:NUM_JOINTS])
        }
        action[GRIPPER_ACTION_KEY] = float(
            np.clip(positions.get("gripper", 50.0), 0.0, 100.0)
        )
        return action

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
        return self._bus is not None and self._bus.is_connected

    @staticmethod
    def joint_keys() -> list[str]:
        """Ordered list of body joint keys (no gripper)."""
        return list(JOINT_ACTION_KEYS)

    @staticmethod
    def gripper_key() -> str:
        """Key for gripper value in the action dict."""
        return GRIPPER_ACTION_KEY


class LerobotSO100LeaderArm(SO100LeaderArm):
    """Backward-compatible alias for existing imports."""
