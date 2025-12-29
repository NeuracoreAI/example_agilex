"""Configuration parameters for Piper robot demos."""

from pathlib import Path

URDF_PATH = str(
    Path(__file__).parent.parent.parent
    / "piper_description"
    / "urdf"
    / "piper_description.urdf"
)

GRIPPER_FRAME_NAME = "gripper_center"


# Pink IK parameters
SOLVER_NAME = "quadprog"
POSITION_COST = 1.0
ORIENTATION_COST = 0.75
FRAME_TASK_GAIN = 0.4
LM_DAMPING = 0.0
DAMPING_COST = 0.25
SOLVER_DAMPING_VALUE = 1e-12

# Controller 1€ Filter parameters
CONTROLLER_MIN_CUTOFF = 0.8  # Minimum cutoff frequency (stabilizes when holding still)
CONTROLLER_BETA = 5.0  # Speed coefficient (reduces lag when moving)
CONTROLLER_D_CUTOFF = 0.9  # Derivative cutoff frequency

# Controller parameters
GRIP_THRESHOLD = 0.9  # Grip value threshold to activate control

# Scaling factors for translation and rotation
TRANSLATION_SCALE = 3.0
ROTATION_SCALE = 2.0

# Thread rates (Hz)
CONTROLLER_DATA_RATE = 50.0  # Controller input reading
IK_SOLVER_RATE = 500.0  # IK solving and robot commands
VISUALIZATION_RATE = 60.0  # GUI updates
ROBOT_RATE = 100.0

# NeuraCore data collection rates
JOINT_STATE_STREAMING_RATE = 100.0  # Data collection rate for neuracore
CAMERA_FRAME_STREAMING_RATE = 60.0  # Data collection rate for camera frame

# # Initial neutral pose for robot (degrees)
NEUTRAL_JOINT_ANGLES = [-5.251, 21.356, -41.386, -4.323, 53.374, 0.0]

# Posture task cost vector (one weight per joint)
POSTURE_COST_VECTOR = [0.0, 0.0, 0.0, 0.05, 0.0, 0.0]


POLICY_EXECUTION_RATE = 100.0  # Hz
PREDICTION_HORIZON_EXECUTION_RATIO = (
    0.8  # percentage of the prediction horizon that is executed
)
MAX_SAFETY_THRESHOLD = 20.0  # degrees
MAX_ACTION_ERROR_THRESHOLD = 3.0  # degrees
TARGETING_POSE_TIME_THRESHOLD = 1.0  # seconds

GRIPPER_LOGGING_NAME = "gripper"
CAMERA_LOGGING_NAME = "rgb"
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
