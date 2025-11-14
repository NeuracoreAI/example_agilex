#!/usr/bin/env python3
"""Piper Robot Teleoperation with Meta Quest Controller - REAL ROBOT CONTROL.

This demo uses Pink IK control with Meta Quest controller input to control the REAL robot.
- REAL ROBOT CONTROL - sends commands to physical robot!
- Uses right hand controller grip as dead man's button
- Uses ROS pointer frame for natural pointing control
- Applies relative transformations accounting for different coordinate frames
"""

import sys
import threading
import time
import traceback
from enum import Enum
from pathlib import Path

import numpy as np

# Add parent directory to path to import pink_ik_solver and piper_controller
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add meta_quest_reader to path
sys.path.insert(0, str(Path(__file__).parent.parent / "meta_quest_reader"))


import viser
import yourdfpy
from configs import (
    DAMPING_COST,
    DATA_COLLECTION_RATE,
    FRAME_TASK_GAIN,
    GRIP_THRESHOLD,
    GRIPPER_FRAME_NAME,
    IK_SOLVER_RATE,
    LM_DAMPING,
    NEUTRAL_JOINT_ANGLES,
    ORIENTATION_COST,
    POSITION_COST,
    POSTURE_COST_VECTOR,
    ROBOT_RATE,
    SMOOTHING_ALPHA,
    SOLVER_DAMPING_VALUE,
    SOLVER_NAME,
    URDF_PATH,
    VISUALIZATION_RATE,
)
from meta_quest_reader.reader import MetaQuestReader
from scipy.spatial.transform import Rotation, Slerp
from viser.extras import ViserUrdf

from pink_ik_solver import PinkIKSolver
from piper_controller import PiperController


class RobotActivityState(Enum):
    """Robot activity state enumeration."""

    ENABLED = "ENABLED"
    HOMING = "HOMING"
    DISABLED = "DISABLED"


class TeleopState:
    """Thread-safe shared state for teleoperation system.

    This class manages shared data between threads:
    - Data collection thread: updates controller data
    - IK solver thread: reads controller data, updates joint solutions
    - Main thread: reads everything for visualization
    """

    def __init__(self) -> None:
        """Initialize TeleopState with default values."""
        self.lock = threading.Lock()

        # Smoothing parameters
        self.smoothing_alpha: float = (
            1.0  # EMA smoothing factor (0.0-1.0, higher = less smoothing)
        )
        self.controller_transform_raw: np.ndarray | None = None
        self.controller_transform: np.ndarray | None = None  # Smoothed transform

        # Controller data (updated by data collection thread)
        self.grip_value: float = 0.0
        self.trigger_value: float = 0.0

        # Teleoperation state (managed by data collection thread)
        self.teleop_active: bool = False
        # Initial controller and robot transforms when teleop starts (used to calculate delta transform)
        # These are set on rising edge of grip button and reset on falling edge of grip button.
        self.controller_initial_transform: np.ndarray | None = None
        self.robot_initial_transform: np.ndarray | None = None

        # Robot activity state (managed by main thread and data collection thread)
        self.robot_activity_state: RobotActivityState = RobotActivityState.DISABLED

        # IK solution (updated by IK solver thread)
        self.current_joint_config: np.ndarray = NEUTRAL_JOINT_ANGLES.copy()
        self.ik_solve_time_ms: float = 0.0
        self.ik_success: bool = True

        # Target/goal transform (for visualization)
        self.target_transform: np.ndarray | None = None

        # Shutdown flag
        self._shutdown_requested: bool = False

    def get_controller_data(self) -> tuple[np.ndarray | None, float, float]:
        """Get current controller data (thread-safe).

        Returns:
            Tuple of (controller_transform, grip_value, trigger_value)
        """
        with self.lock:
            return (
                (
                    self.controller_transform.copy()
                    if self.controller_transform is not None
                    else None
                ),
                self.grip_value,
                self.trigger_value,
            )

    def set_controller_data(
        self, transform: np.ndarray, grip: float, trigger: float
    ) -> None:
        """Set controller data (thread-safe).

        Args:
            transform: np.ndarray - 4x4 transformation matrix
            grip: float - grip value
            trigger: float - trigger value

        Raises:
            ValueError: If the transform is not a 4x4 matrix
            ValueError: If the grip value is not between 0.0 and 1.0
            ValueError: If the trigger value is not between 0.0 and 1.0
        """
        if transform is not None and transform.shape != (4, 4):
            raise ValueError("Transform must be a 4x4 matrix")
        if grip < 0.0 or grip > 1.0:
            raise ValueError("Grip value must be between 0.0 and 1.0")
        if trigger < 0.0 or trigger > 1.0:
            raise ValueError("Trigger value must be between 0.0 and 1.0")

        with self.lock:
            self.grip_value = grip
            self.trigger_value = trigger

            if transform is not None:
                # Store raw transform
                self.controller_transform_raw = transform.copy()

                # Apply smoothing
                if self.controller_transform is None:
                    # First frame - no smoothing
                    self.controller_transform = transform.copy()
                else:
                    # Apply EMA smoothing to transform
                    self.controller_transform = self._smooth_transform(
                        self.controller_transform,  # previous (smoothed)
                        transform,  # new (raw)
                        self.smoothing_alpha,
                    )
            else:
                self.controller_transform = None
                self.controller_transform_raw = None

    def set_teleop_state(
        self,
        active: bool,
        controller_initial: np.ndarray | None,
        robot_initial: np.ndarray | None,
    ) -> None:
        """Set teleoperation state (thread-safe).

        Args:
            active: bool - whether teleop is active
            controller_initial: np.ndarray | None - 4x4 transformation matrix for initial controller transform or None to clear
            robot_initial: np.ndarray | None - 4x4 transformation matrix for initial robot transform or None to clear
        """
        with self.lock:
            self.teleop_active = active
            self.controller_initial_transform = (
                controller_initial.copy() if controller_initial is not None else None
            )
            self.robot_initial_transform = (
                robot_initial.copy() if robot_initial is not None else None
            )

    def get_teleop_active(self) -> bool:
        """Get teleoperation active state (thread-safe)."""
        with self.lock:
            return self.teleop_active

    def get_initial_robot_controller_transforms(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Get initial robot and controller transforms.

        These two transforms are captured on rising edge of grip button
        and reset on falling edge of grip button. (thread-safe)

        Returns:
            Tuple of (controller_initial_transform, robot_initial_transform)
        """
        with self.lock:
            return (
                (
                    self.controller_initial_transform.copy()
                    if self.controller_initial_transform is not None
                    else None
                ),
                (
                    self.robot_initial_transform.copy()
                    if self.robot_initial_transform is not None
                    else None
                ),
            )

    def get_robot_activity_state(self) -> RobotActivityState:
        """Get robot activity state (thread-safe).

        Returns:
            RobotActivityState - current robot activity state
        """
        with self.lock:
            return self.robot_activity_state

    def set_robot_activity_state(self, state: RobotActivityState) -> None:
        """Set robot activity state (thread-safe).

        Args:
            state: RobotActivityState - new robot activity state
        """
        with self.lock:
            self.robot_activity_state = state

    def get_joint_config(self) -> np.ndarray:
        """Get current joint configuration (thread-safe).

        Returns:
            Current joint configuration
        """
        with self.lock:
            return self.current_joint_config.copy()

    def set_joint_config(self, config: np.ndarray) -> None:
        """Set joint configuration (thread-safe).

        Args:
            config: np.ndarray - joint configuration
        """
        with self.lock:
            self.current_joint_config = config.copy()

    def set_target_transform(self, transform: np.ndarray | None) -> None:
        """Set target transform for visualization (thread-safe).

        Args:
            transform: np.ndarray | None - 4x4 transformation matrix or None to clear target transform
        """
        with self.lock:
            self.target_transform = transform.copy() if transform is not None else None

    def get_target_transform(self) -> np.ndarray | None:
        """Get target transform for visualization (thread-safe).

        Returns:
            Target transform or None if target transform is not set
        """
        with self.lock:
            return (
                self.target_transform.copy()
                if self.target_transform is not None
                else None
            )

    def set_ik_solve_time_ms(self, time_ms: float) -> None:
        """Set IK solve time (thread-safe).

        Args:
            time_ms: float - IK solve time in milliseconds
        """
        with self.lock:
            self.ik_solve_time_ms = time_ms

    def set_ik_success(self, success: bool) -> None:
        """Set IK success (thread-safe).

        Args:
            success: bool - True if IK was successful, False otherwise
        """
        with self.lock:
            self.ik_success = success

    def get_ik_solve_time_ms(self) -> float:
        """Get IK solve time (thread-safe).

        Returns:
            IK solve time in milliseconds
        """
        with self.lock:
            return self.ik_solve_time_ms

    def get_ik_success(self) -> bool:
        """Get IK success (thread-safe).

        Returns:
            True if IK was successful, False otherwise
        """
        with self.lock:
            return self.ik_success

    def request_shutdown(self) -> None:
        """Request shutdown of all threads (thread-safe)."""
        with self.lock:
            self._shutdown_requested = True

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown is requested (thread-safe).

        Returns:
            True if shutdown is requested, False otherwise
        """
        with self.lock:
            return self._shutdown_requested

    def _smooth_transform(
        self, prev_transform: np.ndarray, new_transform: np.ndarray, alpha: float
    ) -> np.ndarray:
        """Apply exponential moving average smoothing to a transform matrix.

        Args:
            prev_transform: np.ndarray - Previous (smoothed) transform matrix (4x4)
            new_transform: np.ndarray - New (raw) transform matrix (4x4)
            alpha: float - Smoothing factor (0.0-1.0), higher = less smoothing

        Returns:
            np.ndarray - Smoothed transform matrix (4x4)
        """
        # Smooth position using EMA
        prev_pos = prev_transform[:3, 3]
        new_pos = new_transform[:3, 3]
        smoothed_pos = (1 - alpha) * prev_pos + alpha * new_pos

        # Smooth orientation using SLERP (quaternion spherical interpolation)
        prev_rot = Rotation.from_matrix(prev_transform[:3, :3])
        new_rot = Rotation.from_matrix(new_transform[:3, :3])

        # SLERP interpolation using SciPy's Slerp class
        key_times = [0.0, 1.0]
        key_rots = Rotation.concatenate([prev_rot, new_rot])
        slerp = Slerp(key_times, key_rots)
        smoothed_rot = slerp([alpha])[0]

        # Build smoothed transform
        smoothed_transform = np.eye(4)
        smoothed_transform[:3, 3] = smoothed_pos
        smoothed_transform[:3, :3] = smoothed_rot.as_matrix()

        return smoothed_transform

    def set_smoothing_alpha(self, alpha: float) -> None:
        """Update smoothing factor (thread-safe).

        Args:
            alpha: float - smoothing factor (0.0-1.0), higher = less smoothing
        """
        with self.lock:
            self.smoothing_alpha = alpha


# TODO: turn state.is_shutdown_requested into a thread event
def data_collection_thread(
    state: TeleopState, quest_reader: MetaQuestReader, ik_solver: PinkIKSolver
) -> None:
    """Data collection thread - reads controller data and manages teleop state.

    This thread runs at high frequency to ensure responsive controller input.
    Handles:
    - Reading Meta Quest controller data
    - Processing grip button (dead man's switch)
    - Managing teleop activation/deactivation
    - Capturing initial poses when teleop activates

    Args:
        state: TeleopState object for thread-safe communication
        quest_reader: MetaQuestReader instance
        ik_solver: PinkIKSolver instance (for getting current end-effector pose)
    """
    print("📡 Data collection thread started")

    dt: float = 1.0 / DATA_COLLECTION_RATE
    prev_grip_active: bool = False

    try:
        while not state.is_shutdown_requested():
            loop_start = time.time()

            # Update Meta Quest data
            quest_reader.update()

            # Get controller data
            grip_value = quest_reader.get_grip_value("right")
            trigger_value = quest_reader.get_trigger_value("right")
            controller_transform = quest_reader.get_hand_controller_transform_ros(
                hand="right"
            )

            # Update shared state with controller data
            state.set_controller_data(controller_transform, grip_value, trigger_value)

            # Grip button logic (dead man's switch)
            robot_activity_state = state.get_robot_activity_state()
            # Teleop can only be activated if robot is ENABLED (not HOMING or DISABLED)
            grip_active = (
                grip_value >= GRIP_THRESHOLD
                and robot_activity_state == RobotActivityState.ENABLED
            )

            # Rising edge - grip just pressed AND robot is enabled
            if (
                grip_active
                and not prev_grip_active
                and controller_transform is not None
            ):
                # Start teleop control
                # capture initial poses
                controller_initial_transform = controller_transform.copy()

                # Capture initial robot end-effector pose
                position, orientation = ik_solver.get_current_end_effector_pose()

                robot_initial_transform = np.eye(4)
                robot_initial_transform[:3, :3] = orientation
                robot_initial_transform[:3, 3] = position

                state.set_teleop_state(
                    True, controller_initial_transform, robot_initial_transform
                )

                print("✓ Teleop control activated")
                print(
                    f"  Controller initial position: {controller_initial_transform[:3, 3]}"
                )
                print(f"  Robot initial position: {robot_initial_transform[:3, 3]}")

            # Falling edge - grip just released OR robot disabled
            elif not grip_active and prev_grip_active:
                # Stop teleop control
                state.set_teleop_state(False, None, None)
                print("✗ Teleop control deactivated")

            prev_grip_active = grip_active

            # Sleep to maintain loop rate (check shutdown more frequently)
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"❌ Data collection thread error: {e}")
        traceback.print_exc()
        state.request_shutdown()
    finally:
        # Ensure clean exit - deactivate teleop
        state.set_teleop_state(False, None, None)
        print("📡 Data collection thread stopped")


def ik_solver_thread(
    state: TeleopState, ik_solver: PinkIKSolver, robot_controller: PiperController
) -> None:
    """IK solver thread - solves IK and sends commands to robot.

    This thread runs at medium frequency to solve IK and send commands.
    Handles:
    - Computing target transforms from controller deltas
    - Solving inverse kinematics
    - Sending joint commands to robot
    - Sending gripper commands

    Args:
        state: TeleopState - object for thread-safe communication
        ik_solver: PinkIKSolver - instance
        robot_controller: PiperController - instance

    Raises:
        Exception: If an error occurs during the thread execution
    """
    print("🧮 IK solver thread started")

    dt: float = 1.0 / IK_SOLVER_RATE

    try:
        while not state.is_shutdown_requested():
            loop_start: float = time.time()

            # Get current state
            # controller transform to control the robot, trigger value to open/close gripper
            controller_transform, _, trigger_value = state.get_controller_data()
            teleop_active = state.get_teleop_active()
            controller_initial, robot_initial = (
                state.get_initial_robot_controller_transforms()
            )
            robot_activity_state = state.get_robot_activity_state()

            # During homing, track robot's actual position for visualization
            if robot_activity_state == RobotActivityState.HOMING:
                # Read actual robot joint angles
                current_joint_angles = robot_controller.get_current_joint_angles()
                if current_joint_angles is not None:
                    # Convert to radians and update IK solver configuration
                    current_joint_angles_rad = np.radians(current_joint_angles)
                    ik_solver.set_configuration(current_joint_angles_rad)

                    # Update shared state for visualization
                    state.set_joint_config(current_joint_angles_rad)

                    # Update target transform based on current end effector pose
                    position, orientation = ik_solver.get_current_end_effector_pose()
                    current_target_transform = np.eye(4)
                    current_target_transform[:3, :3] = orientation
                    current_target_transform[:3, 3] = position
                    state.set_target_transform(current_target_transform)

                    # Skip IK solving during homing since we're just tracking
                    state.set_ik_success(True)
                    state.set_ik_solve_time_ms(0.0)
                else:
                    # If we can't read joint angles, keep current config
                    joint_config = ik_solver.get_current_configuration()
                    state.set_joint_config(joint_config)
                    state.set_ik_success(False)

                # Skip the rest of the loop iteration during homing
                elapsed = time.time() - loop_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            # Get target transform for IK solving if teleop is active and not homing
            T_robot_target = None
            if (
                teleop_active
                and not robot_activity_state == RobotActivityState.HOMING
                and controller_transform is not None
                and controller_initial is not None
                and robot_initial is not None
            ):
                # TODO: figure out if this is the correct option from the three options in the browser demo
                # Calculate delta transform in controller space as translation and rotation matrix
                delta_position = controller_transform[:3, 3] - controller_initial[:3, 3]
                delta_orientation = (
                    controller_transform[:3, :3] @ controller_initial[:3, :3].T
                )

                T_robot_target = np.eye(4)
                T_robot_target[:3, 3] = robot_initial[:3, 3] + delta_position
                # Apply the orientation delta to the initial robot orientation
                T_robot_target[:3, :3] = delta_orientation @ robot_initial[:3, :3]

            # Solve IK
            if T_robot_target is not None:
                # set target pose for IK solver
                ik_solver.set_target_pose(T_robot_target[:3, 3], T_robot_target[:3, :3])
                # Store target transform for visualization
                state.set_target_transform(T_robot_target)
            else:
                # No target - clear visualization
                state.set_target_transform(None)

            success = ik_solver.solve_ik()

            if success:
                # Get joint configuration (IK solution)
                joint_config = ik_solver.get_current_configuration()

                # Update timing
                stats = ik_solver.get_statistics()
                solve_time_ms = stats["last_solve_time_ms"]

                # Update shared state with IK solution for the robot controller to use and for visualization
                state.set_joint_config(joint_config)
                # Update timing for visualization
                state.set_ik_solve_time_ms(solve_time_ms)
                state.set_ik_success(success)

                # Send to REAL ROBOT only if ENABLED (not HOMING or DISABLED)
                if robot_activity_state == RobotActivityState.ENABLED:
                    try:
                        # Send joint angles
                        robot_joint_angles = np.degrees(joint_config[:6])
                        robot_controller.set_target_joint_angles(
                            robot_joint_angles.tolist()
                        )
                        # Send gripper command only if teleop is active
                        if teleop_active:
                            # Send gripper command based on trigger value
                            # Trigger: 0.0 = open gripper, 1.0 = closed gripper
                            # Gripper open value: 101.0 = fully open, 1.0 = fully closed
                            # Invert mapping: trigger 0.0 -> gripper 101.0, trigger 1.0 -> gripper 1.0
                            gripper_open_value = (1.0 - trigger_value) * 100.0 + 1.0
                            robot_controller.set_gripper_open_value(gripper_open_value)

                    except Exception as e:
                        print(f"Failed to send joint command: {e}")
                        state.set_robot_activity_state(RobotActivityState.DISABLED)
            else:
                # IK failed
                joint_config = ik_solver.get_current_configuration()
                state.set_joint_config(joint_config)
                state.set_ik_solve_time_ms(0.0)
                state.set_ik_success(False)

            # Sleep to maintain loop rate
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"❌ IK solver thread error: {e}")
        import traceback

        traceback.print_exc()
        state.request_shutdown()
    finally:
        print("🧮 IK solver thread stopped")


# Configuration loaded from configs.py


print("=" * 60)
print("PIPER ROBOT TELEOPERATION - REAL ROBOT CONTROL")
print("=" * 60)
print("Thread frequencies:")
print(f"  📡 Data Collection:  {DATA_COLLECTION_RATE} Hz")
print(f"  🧮 IK Solver:        {IK_SOLVER_RATE} Hz")
print(f"  🖥️ Visualization:    {VISUALIZATION_RATE} Hz (running in the main thread)")
print(f"  🤖 Robot Controller: {ROBOT_RATE} Hz")


# Initialize Meta Quest reader
print("\n🎮 Initializing Meta Quest reader...")
ip_address = sys.argv[1] if len(sys.argv) > 1 else None
quest_reader = MetaQuestReader(
    ip_address=ip_address, port=5555, print_FPS=False, run=True
)

# Initialize robot controller
print("\n🤖 Initializing Piper robot controller...")
robot_controller = PiperController(
    can_interface="can0",
    robot_rate=ROBOT_RATE,
    control_mode=PiperController.ControlMode.JOINT_SPACE,
    debug_mode=False,
)

# set initial configuration to current joint angles
initial_joint_angles = np.radians(robot_controller.get_current_joint_angles())
if initial_joint_angles is None:
    initial_joint_angles = np.radians(NEUTRAL_JOINT_ANGLES)

# Create Pink IK solver
print("\n🔧 Creating Pink IK solver...")
ik_solver = PinkIKSolver(
    urdf_path=URDF_PATH,
    end_effector_frame=GRIPPER_FRAME_NAME,
    solver_name=SOLVER_NAME,
    position_cost=POSITION_COST,
    orientation_cost=ORIENTATION_COST,
    frame_task_gain=FRAME_TASK_GAIN,
    lm_damping=LM_DAMPING,
    damping_cost=DAMPING_COST,
    solver_damping_value=SOLVER_DAMPING_VALUE,
    integration_time_step=1 / IK_SOLVER_RATE,
    initial_configuration=initial_joint_angles,
    posture_cost_vector=np.array(POSTURE_COST_VECTOR),
)

# Initialize shared state
state = TeleopState()
state.set_smoothing_alpha(SMOOTHING_ALPHA)

# Set up visualizer
print("\n🖥️  Starting visualization...")
server = viser.ViserServer()
server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

# Load URDF for visualization
urdf = yourdfpy.URDF.load(URDF_PATH)

# Robot visualization - shows IK solution (commanded position)
urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

# Create transform control for controller visualization
controller_handle = server.scene.add_transform_controls(
    "/controller",
    scale=0.15,
    position=(0, 0, 0),
    wxyz=(1, 0, 0, 0),
)

# Create coordinate frame for target/goal visualization
target_frame_handle = server.scene.add_frame(
    "/target_goal", axes_length=0.1, axes_radius=0.003
)

# Add GUI controls
timing_handle = server.gui.add_number("IK Solve Time (ms)", 0.001, disabled=True)
grip_value_handle = server.gui.add_number("Grip Value", 0.0, disabled=True)
trigger_value_handle = server.gui.add_number(
    "Trigger Value (Gripper)", 0.0, disabled=True
)
teleop_status_handle = server.gui.add_text("Teleop Status", "Inactive")
controller_status_handle = server.gui.add_text("Controller Status", "Waiting...")
joint_angles_handle = server.gui.add_text("Joint Angles (IK Solution)", "Waiting...")
robot_status_handle = server.gui.add_text("Robot Status", "Initializing...")
gripper_status_handle = server.gui.add_text("Gripper Status", "Open (0%)")

# Homing controls
homing_status_handle = server.gui.add_text("Homing Status", "Idle")
go_home_button = server.gui.add_button("Go Home")

# Robot control buttons
enable_robot_handle = server.gui.add_button("Enable Robot")
disable_robot_handle = server.gui.add_button("Disable Robot")
emergency_stop_handle = server.gui.add_button("Emergency Stop")

# Smoothing control
smoothing_alpha_handle = server.gui.add_number(
    "Smoothing Factor", SMOOTHING_ALPHA, min=0.0, max=1.0, step=0.01
)

# Pink parameters
position_weight_handle = server.gui.add_number(
    "Position Weight", POSITION_COST, min=0.0, max=10.0, step=0.1
)
orientation_weight_handle = server.gui.add_number(
    "Orientation Weight", ORIENTATION_COST, min=0.0, max=1.0, step=0.01
)
frame_task_gain_handle = server.gui.add_number(
    "Frame Task Gain", FRAME_TASK_GAIN, min=0.0, max=10.0, step=0.1
)
lm_damping_handle = server.gui.add_number(
    "LM Damping", LM_DAMPING, min=0.0, max=5.0, step=0.01
)
damping_weight_handle = server.gui.add_number(
    "Damping Weight", DAMPING_COST, min=0.0, max=1.0, step=0.01
)
solver_damping_value_handle = server.gui.add_number(
    "Solver Damping Value", SOLVER_DAMPING_VALUE, min=0.0, max=1.0, step=0.0001
)

# Posture cost controls (one per joint)
posture_cost_handles = []
for i in range(len(POSTURE_COST_VECTOR)):
    handle = server.gui.add_number(
        f"Posture Cost J{i+1}", POSTURE_COST_VECTOR[i], min=0.0, max=1.0, step=0.01
    )
    posture_cost_handles.append(handle)

# Start robot control loop
print("\n🚀 Starting robot control loop...")
robot_controller.start_control_loop()

# Start data collection thread
print("\n📡 Starting data collection thread...")
data_thread = threading.Thread(
    target=data_collection_thread, args=(state, quest_reader, ik_solver), daemon=True
)
data_thread.start()

# Start IK solver thread
print("\n🧮 Starting IK solver thread...")
ik_thread = threading.Thread(
    target=ik_solver_thread, args=(state, ik_solver, robot_controller), daemon=True
)
ik_thread.start()


# Register Button B callback for home position
def on_button_b_pressed() -> None:
    """Handle Button B press to move robot to home position."""
    robot_activity_state = state.get_robot_activity_state()
    if robot_activity_state == RobotActivityState.ENABLED:
        print("🏠 Button B pressed - Moving to home position...")
        # Set state to HOMING to prevent IK thread from sending robot commands
        state.set_robot_activity_state(RobotActivityState.HOMING)
        # Disable teleop during homing
        state.set_teleop_state(False, None, None)
        homing_status_handle.value = "Homing: In progress (Button B)"
        ok = robot_controller.move_to_home()
        if not ok:
            print("✗ Failed to initiate home move")
            homing_status_handle.value = "Homing: Failed"
            # Revert to ENABLED on failure
            state.set_robot_activity_state(RobotActivityState.ENABLED)
        else:
            homing_status_handle.value = "Homing: In progress (Button B)"
    else:
        print("⚠️  Button B pressed but robot is not enabled")
        homing_status_handle.value = "Homing: Robot not enabled"


quest_reader.on("button_b_pressed", on_button_b_pressed)

print()
print("🚀 Starting teleoperation with REAL ROBOT CONTROL...")
print("🎮 CONTROLS:")
print("   1. Click 'Enable Robot' button in GUI first")
print("   2. Hold RIGHT GRIP to activate teleoperation")
print("   3. Move controller - robot follows!")
print("   4. Hold RIGHT TRIGGER to close gripper")
print("   5. Press BUTTON B to send robot home")
print("   6. Release grip to stop")
print("   7. Use 'Emergency Stop' if needed")
print("⚠️  Press Ctrl+C to exit")
print()

# Visualization loop variables
dt: float = 1.0 / VISUALIZATION_RATE

try:
    while True:
        loop_start: float = time.time()

        # Handle GUI buttons (robot control)
        if enable_robot_handle.value:
            enable_robot_handle.value = False
            if robot_controller.resume_robot():
                state.set_robot_activity_state(RobotActivityState.ENABLED)
                print("✓ Robot enabled")
            else:
                print("✗ Failed to enable robot")

        if disable_robot_handle.value:
            disable_robot_handle.value = False
            state.set_robot_activity_state(RobotActivityState.DISABLED)
            robot_controller.graceful_stop()
            # Reset teleop state when disabling robot
            state.set_teleop_state(False, None, None)
            print("✓ Robot disabled")

        if emergency_stop_handle.value:
            emergency_stop_handle.value = False
            state.set_robot_activity_state(RobotActivityState.DISABLED)
            robot_controller.emergency_stop()
            # Reset teleop state on emergency stop
            state.set_teleop_state(False, None, None)
            print("🚨 Emergency stop activated!")

        # Update smoothing factor
        state.set_smoothing_alpha(smoothing_alpha_handle.value)

        # Update Pink parameters (GUI controls)
        # Collect posture cost values from GUI handles
        posture_cost_vector = np.array(
            [handle.value for handle in posture_cost_handles]
        )
        ik_solver.update_task_parameters(
            position_cost=position_weight_handle.value,
            orientation_cost=orientation_weight_handle.value,
            frame_task_gain=frame_task_gain_handle.value,
            lm_damping=lm_damping_handle.value,
            damping_cost=damping_weight_handle.value,
            solver_damping_value=solver_damping_value_handle.value,
            posture_cost_vector=posture_cost_vector,
        )

        # Handle homing button
        if go_home_button.value:
            go_home_button.value = False
            robot_activity_state = state.get_robot_activity_state()
            if robot_activity_state == RobotActivityState.ENABLED:
                print("🏠 GUI: Moving to home position...")
                state.set_robot_activity_state(RobotActivityState.HOMING)
                state.set_teleop_state(False, None, None)
                homing_status_handle.value = "Homing: In progress (GUI)"
                ok = robot_controller.move_to_home()
                if not ok:
                    print("✗ Failed to initiate home move")
                    homing_status_handle.value = "Homing: Failed"
                    # Revert to ENABLED on failure
                    state.set_robot_activity_state(RobotActivityState.ENABLED)
                else:
                    homing_status_handle.value = "Homing: In progress"
            else:
                print("⚠️  Cannot home: robot not enabled")
                homing_status_handle.value = "Homing: Robot not enabled"

        # Check if homing has completed and sync IK solver
        if state.get_robot_activity_state() == RobotActivityState.HOMING:
            # Get current robot joint angles
            current_joint_angles = robot_controller.get_current_joint_angles()
            if current_joint_angles is not None:
                # Convert to radians for comparison
                current_joint_angles_rad = np.radians(current_joint_angles)
                home_joint_angles_rad = np.radians(robot_controller.HOME_JOINT_ANGLES)

                # Check if robot has reached home (within 2 degrees per joint)
                joint_errors = np.abs(current_joint_angles_rad - home_joint_angles_rad)
                max_error_rad = np.deg2rad(2.0)  # 2 degrees tolerance

                if np.all(joint_errors < max_error_rad):
                    # Robot has reached home - sync IK solver
                    print("✓ Robot reached home position - syncing IK solver...")

                    # Update IK solver configuration to match robot's actual position
                    ik_solver.set_configuration(current_joint_angles_rad)

                    # Update target transform to match home position
                    position, orientation = ik_solver.get_current_end_effector_pose()
                    home_target_transform = np.eye(4)
                    home_target_transform[:3, :3] = orientation
                    home_target_transform[:3, 3] = position
                    state.set_target_transform(home_target_transform)

                    # Update shared state with new joint configuration
                    state.set_joint_config(current_joint_angles_rad)

                    # Transition back to ENABLED
                    state.set_robot_activity_state(RobotActivityState.ENABLED)
                    homing_status_handle.value = "Homing: Complete"
                    print("✓ IK solver synced and robot re-enabled")

        # Get data from shared state
        controller_transform, grip_value, trigger_value = state.get_controller_data()
        teleop_active = state.get_teleop_active()
        # Get updated robot_activity_state (may have changed if homing completed)
        robot_activity_state = state.get_robot_activity_state()
        joint_config = state.get_joint_config()
        solve_time_ms = state.get_ik_solve_time_ms()
        ik_success = state.get_ik_success()
        target_transform = state.get_target_transform()

        # Update GUI displays
        grip_value_handle.value = grip_value
        trigger_value_handle.value = trigger_value

        ema_timing: float = 0.001  # Exponential moving average for timing display
        # Update timing display with exponential moving average
        ema_timing = 0.99 * ema_timing + 0.01 * solve_time_ms
        timing_handle.value = ema_timing

        # Update controller visualization if available
        if controller_transform is not None:
            controller_pos = controller_transform[:3, 3]
            controller_rot = Rotation.from_matrix(controller_transform[:3, :3])
            controller_quat_xyzw = controller_rot.as_quat()
            controller_quat_wxyz = [
                controller_quat_xyzw[3],
                controller_quat_xyzw[0],
                controller_quat_xyzw[1],
                controller_quat_xyzw[2],
            ]

            # Set controller visualization to current controller position and orientation
            controller_handle.position = tuple(controller_pos)
            controller_handle.wxyz = tuple(controller_quat_wxyz)

            # Update controller status display
            controller_status_str = "Controller Status:\n"
            controller_status_str += f"  Position: [{controller_pos[0]:.3f}, {controller_pos[1]:.3f}, {controller_pos[2]:.3f}]\n"
            controller_status_str += "  Connected: ✓\n"
            controller_status_handle.value = controller_status_str
        else:
            controller_status_handle.value = "Controller Status:\n  Connected: ✗"

        # Update teleop status display - simple Active/Inactive
        if teleop_active:
            teleop_status_handle.value = "Teleop Status: Active"
        else:
            teleop_status_handle.value = "Teleop Status: Inactive"

        # Update target/goal visualization
        if target_transform is not None:
            target_pos = target_transform[:3, 3]
            target_rot = Rotation.from_matrix(target_transform[:3, :3])
            target_quat_xyzw = target_rot.as_quat()
            target_quat_wxyz = [
                target_quat_xyzw[3],
                target_quat_xyzw[0],
                target_quat_xyzw[1],
                target_quat_xyzw[2],
            ]

            target_frame_handle.position = tuple(target_pos)
            target_frame_handle.wxyz = tuple(target_quat_wxyz)

        # Update robot visualization
        if ik_success:
            # TODO: figure out if we want to visualise the IK solution or the robot's current state
            urdf_vis.update_cfg(joint_config)

            # Update joint angles display
            joint_angles_str = "Joint Angles (IK Solution):\n"
            joint_angles_deg = np.degrees(joint_config)
            for i in range(6):
                angle_rad = joint_config[i]
                angle_deg = joint_angles_deg[i]
                joint_angles_str += (
                    f"  J{i+1}: {angle_rad:.3f} rad ({angle_deg:.1f}°)\n"
                )
            joint_angles_handle.value = joint_angles_str

        # Update robot status - simple Enabled/Homing/Disabled
        if robot_activity_state == RobotActivityState.ENABLED:
            robot_status_handle.value = "Robot Status: Enabled"
        elif robot_activity_state == RobotActivityState.HOMING:
            robot_status_handle.value = "Robot Status: Homing"
        else:  # DISABLED
            robot_status_handle.value = "Robot Status: Disabled"

        # Update gripper status (always show, even if robot disabled)
        # Trigger value: 0.0 = open, 1.0 = closed
        gripper_closed_percent = trigger_value * 100.0
        if trigger_value > 0.9:
            gripper_state = "Closed"
        elif trigger_value > 0.1:
            gripper_state = "Closing"
        else:
            gripper_state = "Open"

        if robot_activity_state == RobotActivityState.ENABLED:
            gripper_status_handle.value = (
                f"Gripper: {gripper_state} ({gripper_closed_percent:.0f}% closed)"
            )
        else:
            gripper_status_handle.value = f"Gripper: {gripper_state} ({gripper_closed_percent:.0f}% closed) [Disabled]"

        # Sleep to maintain visualization rate
        elapsed = time.time() - loop_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n\n👋 Interrupt received - shutting down gracefully...")
except Exception as e:
    print(f"\n❌ Demo error: {e}")
    import traceback

    traceback.print_exc()

# Cleanup (outside try/except so it always runs)
print("\n🧹 Cleaning up...")

state.request_shutdown()
state.set_robot_activity_state(RobotActivityState.DISABLED)
data_thread.join()
quest_reader.stop()
ik_thread.join()
robot_controller.cleanup()
server.stop()

print("\n👋 Demo stopped.")
