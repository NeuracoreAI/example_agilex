"""Configuration parameters for AgileX Piper robot demos."""

from pathlib import Path

# Hardware & Kinematics Paths
URDF_PATH = str(
    Path(__file__).parent.parent.parent
    / "piper_description"
    / "urdf"
    / "piper_description.urdf"
)
GRIPPER_FRAME_NAME = "gripper_center"

# IK Solver
SOLVER_NAME = "quadprog"

# Teleoperation Thresholds
GRIP_THRESHOLD = 0.9

# Thread Execution Rates (Hz)
CONTROLLER_DATA_RATE = 50.0
IK_SOLVER_RATE = 100
VISUALIZATION_RATE = 60.0
ROBOT_RATE = 100.0
JOINT_STATE_STREAMING_RATE = 100.0
CAMERA_FRAME_STREAMING_RATE = 30.0

# Hardware Context
META_QUEST_AXIS_MASK = [1, 1, 1, 1, 1, 1]
CAMERA_DEVICE_INDEX = 5
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_NAMES = ["rgb_scene", "rgb_wrist"]

# Default Robot Poses
NEUTRAL_JOINT_ANGLES = [-1.6, 52.2, -54.3, -3.2, 43.1, 4.7]
NEUTRAL_END_EFFECTOR_POSE = [455.257, -46.344, 172.213, 176.205, -14.545, 29.621]
# NEUTRAL_JOINT_ANGLES = [-9.3, 86.7, -86.6, 1.8, 61.7, -5.1] #Lemon pick task pruthvi

# AI Policy Execution Parameters
POLICY_EXECUTION_RATE = 20.0
PREDICTION_HORIZON_EXECUTION_RATIO = 1.0
MAX_SAFETY_THRESHOLD = 200.0
MAX_ACTION_ERROR_THRESHOLD = 3.0
TARGETING_POSE_TIME_THRESHOLD = 1.0

GRIPPER_NAME = "gripper"
GRIPPER_LOGGING_NAME = GRIPPER_NAME
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
