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
LM_DAMPING = 0.01 #0.0
DAMPING_COST = 0.25
SOLVER_DAMPING_VALUE = 1e-4 #1e-12

# Controller 1€ Filter parameters
CONTROLLER_MIN_CUTOFF = 0.8  # Minimum cutoff frequency (stabilizes when holding still)
CONTROLLER_BETA = 0.05 #5.0  # Speed coefficient (reduces lag when moving)
CONTROLLER_D_CUTOFF = 0.9  # Derivative cutoff frequency

# Controller parameters
GRIP_THRESHOLD = 0.9  # Grip value threshold to activate control

# Scaling factors for translation and rotation
TRANSLATION_SCALE = 1.5
ROTATION_SCALE = 1.2
SLOW_TRANSLATION_SCALE = 0.6
SLOW_ROTATION_SCALE = 0.6
WRIST_JOINT_BUTTON_STEP_DEGREES = 5.0

# Thread rates (Hz)
CONTROLLER_DATA_RATE = 50.0  # Controller input reading
IK_SOLVER_RATE = 100 #250.0  # IK solving and robot commands
VISUALIZATION_RATE = 60.0  # GUI updates
ROBOT_RATE = 100.0

# Neuracore data collection rates
JOINT_STATE_STREAMING_RATE = 100.0  # Data collection rate for neuracore
CAMERA_FRAME_STREAMING_RATE = 30.0  # Data collection rate for camera frame

# Meta quest axis mask
META_QUEST_AXIS_MASK = [1, 1, 1, 1, 1, 1]  # [x, y, z, roll, pitch, yaw]

# USB webcam (OpenCV) configuration
CAMERA_DEVICE_INDEX = 5  # 0 = first camera, 1 = second, etc.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# # Initial neutral pose for robot (mm and degrees)
# NEUTRAL_JOINT_ANGLES = [-1.003, 80.167, -51.064, -4.127, 16.548, 2.619]
# NEUTRAL_JOINT_ANGLES = [0.4, 112.3, -101.0, -2.6, 63.2, 18.3]
#NEUTRAL_JOINT_ANGLES = [-9.3, 86.7, -86.6, 1.8, 61.7, -5.1] #Lemon pick task pruthvi
NEUTRAL_JOINT_ANGLES = [-1.6, 52.2, -54.3, -3.2, 43.1, 4.7]  # halid
# Joint Angles (rad):  J1: -0.028 rad (-1.6°)  J2: 0.910 rad (52.2°)  J3: -0.948 rad (-54.3°)  J4: -0.057 rad (-3.2°)  J5: 0.752 rad (43.1°)  J6: 0.083 rad (4.7°)
                                                                                                                                                           
NEUTRAL_END_EFFECTOR_POSE = [455.257, -46.344, 172.213, 176.205, -14.545, 29.621]

# Posture task cost vector (one weight per joint)
POSTURE_COST_VECTOR = [0.0, 0.0, 0.0, 0.05, 0.0, 0.0]


POLICY_EXECUTION_RATE = 20.0  # Hz
PREDICTION_HORIZON_EXECUTION_RATIO = (
    1.0  # percentage of the prediction horizon that is executed
)
MAX_SAFETY_THRESHOLD = 200.0  # degrees
MAX_ACTION_ERROR_THRESHOLD = 3.0  # degrees
TARGETING_POSE_TIME_THRESHOLD = 1.0  # seconds

GRIPPER_NAME = "gripper"
GRIPPER_LOGGING_NAME = GRIPPER_NAME

# Camera logging names for Neuracore (scene + wrist)
CAMERA_NAMES = ["rgb_scene", "rgb_wrist"]

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
