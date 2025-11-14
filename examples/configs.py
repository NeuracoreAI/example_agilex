"""Configuration parameters for Piper robot demos."""

from pathlib import Path

URDF_PATH = str(
    Path(__file__).parent.parent
    / "piper_description"
    / "urdf"
    / "piper_description.urdf"
)
GRIPPER_FRAME_NAME = "gripper_center"

# Pink IK parameters
SOLVER_NAME = "quadprog"
POSITION_COST = 1.0
ORIENTATION_COST = 0.5
FRAME_TASK_GAIN = 0.5
LM_DAMPING = 0.0
DAMPING_COST = 0.15
SMOOTHING_ALPHA = 0.4
SOLVER_DAMPING_VALUE = 1e-12

# Controller parameters
GRIP_THRESHOLD = 0.9  # Grip value threshold to activate control

# Thread rates (Hz)
DATA_COLLECTION_RATE = 50.0  # Controller input reading
IK_SOLVER_RATE = 500.0  # IK solving and robot commands
VISUALIZATION_RATE = 60.0  # GUI updates
ROBOT_RATE = 100.0

# Initial neutral pose for robot (radians)
NEUTRAL_JOINT_ANGLES = [
    0.0
] * 6  # when using a real piper robot, copy the HOME_JOINT_ANGLES from PiperController class into this list

# Posture task cost vector (one weight per joint)
POSTURE_COST_VECTOR = [0.0, 0.0, 0.0, 0.1, 0.0, 0.0]
