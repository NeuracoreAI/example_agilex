"""Pink URDF Inverse Kinematics Solver.

A generic class for performing inverse kinematics using Pink (Python inverse
kinematics based on Pinocchio) with any URDF robot model.
"""

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pink
import pinocchio as pin
from pink.tasks import DampingTask, FrameTask
from scipy.spatial.transform import Rotation


class PinkIKSolver:
    """A generic Pink-based inverse kinematics solver for URDF robots.

    This class provides a clean interface for setting up and solving inverse
    kinematics problems using Pink with configurable tasks, limits, and
    parameters.
    """

    def __init__(
        self,
        urdf_path: str,
        end_effector_frame: str,
        solver_name: str = "quadprog",
        position_cost: float = 1.0,
        orientation_cost: float = 0.75,
        frame_task_gain: float = 1.0,
        lm_damping: float = 0.25,
        damping_cost: float = 0.0,
        solver_damping_value: float = 1e-12,
        integration_time_step: float = 0.001,
        initial_configuration: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the Pink IK solver.

        Args:
            urdf_path: str - Path to the URDF file
            end_effector_frame: str - Name of the end effector frame
            solver_name: str - Name of the QP solver to use
            position_cost: float - Cost weight for position tracking
            orientation_cost: float - Cost weight for orientation tracking
            frame_task_gain: float - Gain for the frame task
            lm_damping: float - Levenberg-Marquardt damping factor
            damping_cost: float - Cost weight for velocity damping
            solver_damping_value: float - Value for solver damping - Tikhonov regularization parameter
            integration_time_step: float - Time step for integration
            initial_configuration: np.ndarray - Initial joint configuration (if None, uses neutral)
        """
        self.urdf_path: str = urdf_path
        self.end_effector_frame: str = end_effector_frame
        self.solver_name: str = solver_name

        # Task parameters
        self.position_cost: float = position_cost
        self.orientation_cost: float = orientation_cost
        self.frame_task_gain: float = frame_task_gain
        self.lm_damping: float = lm_damping
        self.damping_cost: float = damping_cost
        self.solver_damping_value: float = solver_damping_value
        self.integration_time_step: float = integration_time_step

        # Robot model and data
        self.urdf_model: pin.Model = None
        self.urdf_model_data: pin.ModelData = None
        self.configuration: pink.Configuration = None

        # Tasks
        self.ee_task: FrameTask = None
        self.damping_task: DampingTask = None

        # Statistics
        self.solve_times: list[float] = []
        self.last_solve_time: float = 0.0

        # Initial configuration
        self.initial_configuration: Optional[np.ndarray] = initial_configuration

        # Initialize the robot
        self._build_robot_model()
        self._setup_tasks()

    def _build_robot_model(self) -> None:
        """Build the robot model from URDF."""
        print(f"📁 Loading URDF: {self.urdf_path}")
        self.urdf_model = pin.buildModelFromUrdf(self.urdf_path)
        self.urdf_model_data = self.urdf_model.createData()
        print("✅ Robot loaded successfully!")

        # Validate end effector frame
        self._validate_end_effector_frame()

    def _validate_end_effector_frame(self) -> None:
        """Validate that the end effector frame exists.

        Raises:
            ValueError: If the end effector frame is not found in the URDF.
        """
        try:
            frame_id = self.urdf_model.getFrameId(self.end_effector_frame)
            if frame_id >= self.urdf_model.nframes:
                raise ValueError(f"Frame {self.end_effector_frame} not found in URDF")
        except Exception:
            available_frames = [
                self.urdf_model.frames[i].name for i in range(self.urdf_model.nframes)
            ]
            raise ValueError(
                f"Frame {self.end_effector_frame} not found in URDF. Available frames: {available_frames}"
            )

    def _setup_tasks(self) -> None:
        """Set up Pink tasks and configuration.

        Raises:
            ValueError: If the initial configuration is not valid (initial
                configuration must have the same number of joints as the robot
                model).
        """
        if (
            self.initial_configuration is not None
            and len(self.initial_configuration) != self.urdf_model.nq
        ):
            raise ValueError(
                f"Initial configuration must have {self.urdf_model.nq} values, got {len(self.initial_configuration)}"
            )

        print("🔧 Setting up Pink tasks and limits...")

        self.initial_configuration = (
            self.initial_configuration
            if self.initial_configuration is not None
            else pin.neutral(self.urdf_model)
        )
        self.configuration = pink.Configuration(
            self.urdf_model, self.urdf_model_data, self.initial_configuration
        )

        # Set up end effector task
        self.ee_task = FrameTask(
            self.end_effector_frame,
            position_cost=self.position_cost,
            orientation_cost=self.orientation_cost,
            lm_damping=self.lm_damping,
            gain=self.frame_task_gain,
        )
        self.ee_task.set_target_from_configuration(self.configuration)

        # Set up damping task
        self.damping_task = DampingTask(cost=self.damping_cost)

        print("✅ Tasks configured!")

    def update_task_parameters(
        self,
        position_cost: Optional[float] = None,
        orientation_cost: Optional[float] = None,
        frame_task_gain: Optional[float] = None,
        lm_damping: Optional[float] = None,
        damping_cost: Optional[float] = None,
        solver_damping_value: Optional[float] = None,
        integration_time_step: Optional[float] = None,
    ) -> None:
        """Update task parameters dynamically.

        Args:
            position_cost: float - Cost weight for position tracking
            orientation_cost: float - Cost weight for orientation tracking
            frame_task_gain: float - Gain for the frame task
            lm_damping: float - Levenberg-Marquardt damping factor
            damping_cost: float - Cost weight for velocity damping
            solver_damping_value: float - Value for solver damping - Tikhonov regularization parameter
            integration_time_step: float - Time step for integration
        """
        if position_cost is not None:
            self.position_cost = position_cost
            self.ee_task.position_cost = position_cost

        if orientation_cost is not None:
            self.orientation_cost = orientation_cost
            self.ee_task.orientation_cost = orientation_cost

        if frame_task_gain is not None:
            self.frame_task_gain = frame_task_gain
            self.ee_task.gain = frame_task_gain

        if lm_damping is not None:
            self.lm_damping = lm_damping
            self.ee_task.lm_damping = lm_damping

        if damping_cost is not None:
            self.damping_cost = damping_cost
            self.damping_task.cost = damping_cost

        if solver_damping_value is not None:
            self.solver_damping_value = solver_damping_value

        if integration_time_step is not None:
            self.integration_time_step = integration_time_step

    def set_target_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        """Set target pose from position and orientation.

        Args:
            position: np.ndarray - 3D position vector
            orientation: np.ndarray - 3x3 rotation matrix or 4-element
                quaternion (wxyz)

        Raises:
            ValueError: If the orientation is not a 3x3 matrix or 4-element
                quaternion.
        """
        if orientation.shape == (4,):
            # Quaternion (wxyz) to rotation matrix
            target_rotation = Rotation.from_quat(
                [orientation[1], orientation[2], orientation[3], orientation[0]]
            )
            rotation_matrix = target_rotation.as_matrix()
        elif orientation.shape == (3, 3):
            # Already a rotation matrix
            rotation_matrix = orientation
        else:
            raise ValueError("Orientation must be a 3x3 matrix or 4-element quaternion")

        target_transform = pin.SE3(rotation_matrix, position)
        self.ee_task.set_target(target_transform)

    def solve_ik(self, dt: Optional[float] = None) -> bool:
        """Solve inverse kinematics for current target.

        Args:
            dt: float - Integration time step (uses instance default if None).

        Returns:
            True if successful, False otherwise.

        Raises:
            Exception: If an error occurs during the IK solve.
        """
        if dt is None:
            dt = self.integration_time_step

        start_time = time.time()

        try:
            # Prepare tasks and limits
            tasks = (
                self.ee_task,
                self.damping_task,
            )
            limits = (
                self.configuration.model.configuration_limit,
                self.configuration.model.velocity_limit,
            )

            # Solve differential IK
            velocity = pink.solve_ik(
                self.configuration,
                tasks,
                dt,
                solver=self.solver_name,
                damping=self.solver_damping_value,
                limits=limits,
            )

            # Integrate configuration
            self.configuration.integrate_inplace(velocity, dt)

            # Update timing statistics
            elapsed_time = time.time() - start_time
            self.last_solve_time = elapsed_time * 1000  # Convert to ms
            self.solve_times.append(self.last_solve_time)

            # Keep only last 100 solve times for statistics
            if len(self.solve_times) > 100:
                self.solve_times = self.solve_times[-100:]

            return True

        except Exception as e:
            print(f"❌ IK solve failed: {e}")
            return False

    def get_current_configuration(self) -> np.ndarray:
        """Get current joint configuration (in radians)."""
        return self.configuration.q.copy()

    def get_current_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current end effector pose.

        Returns:
            Tuple of (position, rotation_matrix).
        """
        transform = self.configuration.get_transform_frame_to_world(
            self.end_effector_frame
        )
        return transform.translation.copy(), transform.rotation.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get solver timing statistics (in milliseconds)."""
        if not self.solve_times:
            return {
                "last_solve_time_ms": 0.0,
                "avg_solve_time_ms": 0.0,
                "max_solve_time_ms": 0.0,
            }

        return {
            "last_solve_time_ms": self.last_solve_time,
            "avg_solve_time_ms": np.mean(self.solve_times),
            "max_solve_time_ms": np.max(self.solve_times),
            "solve_count": len(self.solve_times),
        }

    def set_configuration(self, joint_config: np.ndarray) -> None:
        """Set the robot to a specific joint configuration.

        Args:
            joint_config: np.ndarray - Array of joint angles to set

        Raises:
            ValueError: If the joint configuration is not valid (joint
                configuration must have the same number of joints as the robot
                model).
        """
        if len(joint_config) != self.urdf_model.nq:
            raise ValueError(
                f"Joint configuration must have {self.urdf_model.nq} values, got {len(joint_config)}"
            )

        self.configuration.update(joint_config)
        self.ee_task.set_target_from_configuration(self.configuration)

    def reset_to_neutral(self) -> None:
        """Reset robot to neutral configuration."""
        self.set_configuration(pin.neutral(self.urdf_model))

    def _get_available_frames(self) -> list[str]:
        """Get list of available frame names."""
        return [self.urdf_model.frames[i].name for i in range(self.urdf_model.nframes)]

    def get_robot_info(self) -> Dict[str, Any]:
        """Get robot information."""
        return {
            "urdf_path": self.urdf_path,
            "end_effector_frame": self.end_effector_frame,
            "num_joints": self.urdf_model.nq,
            "num_dof": self.urdf_model.nv,
            "num_frames": self.urdf_model.nframes,
            "solver_name": self.solver_name,
            "available_frames": self._get_available_frames(),
        }
