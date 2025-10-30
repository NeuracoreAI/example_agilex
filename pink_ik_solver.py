"""
Pink URDF Inverse Kinematics Solver

A generic class for performing inverse kinematics using Pink (Python inverse kinematics 
based on Pinocchio) with any URDF robot model.
"""

import time
import numpy as np
import pink
from pink.tasks import FrameTask, PostureTask, DampingTask
from pink.limits import AccelerationLimit
import pinocchio as pin
from scipy.spatial.transform import Rotation
from typing import Optional, Tuple, Dict, Any


class PinkIKSolver:
    """
    A generic Pink-based inverse kinematics solver for URDF robots.
    
    This class provides a clean interface for setting up and solving inverse kinematics
    problems using Pink with configurable tasks, limits, and parameters.
    """
    
    def __init__(
        self,
        urdf_path: str,
        end_effector_frame: str,
        solver_name: str = "quadprog",
        position_cost: float = 1.0,
        orientation_cost: float = 0.75,
        lm_damping: float = 0.25,
        posture_cost: float = 1e-3,
        damping_cost: float = 0.0,
        integration_time_step: float = 0.001,
        initial_configuration: Optional[np.ndarray] = None
    ):
        """
        Initialize the Pink IK solver.
        
        Args:
            urdf_path: Path to the URDF file
            end_effector_frame: Name of the end effector frame
            solver_name: Name of the QP solver to use
            position_cost: Cost weight for position tracking
            orientation_cost: Cost weight for orientation tracking
            lm_damping: Levenberg-Marquardt damping factor
            posture_cost: Cost weight for posture regularization
            damping_cost: Cost weight for velocity damping
            integration_time_step: Time step for integration
            initial_configuration: Initial joint configuration (if None, uses neutral)
        """
        self.urdf_path = urdf_path
        self.end_effector_frame = end_effector_frame
        self.solver_name = solver_name
        
        # Task parameters
        self.position_cost = position_cost
        self.orientation_cost = orientation_cost
        self.lm_damping = lm_damping
        self.posture_cost = posture_cost
        self.damping_cost = damping_cost
        self.integration_time_step = integration_time_step
        
        # Robot model and data
        self.urdf_model = None
        self.urdf_model_data = None
        self.configuration = None
        
        # Tasks
        self.ee_task = None
        self.posture_task = None
        self.damping_task = None
        
        # Statistics
        self.solve_times = []
        self.last_solve_time = 0.0
        
        # Initialize the robot
        self._build_robot_model()
        self._setup_tasks(initial_configuration)
        
    def _build_robot_model(self):
        """Build the robot model from URDF."""
        print(f"📁 Loading URDF: {self.urdf_path}")
        self.urdf_model = pin.buildModelFromUrdf(self.urdf_path)
        self.urdf_model_data = self.urdf_model.createData()
        print(f"✅ Robot loaded successfully!")
        
        # Validate end effector frame
        self._validate_end_effector_frame()
        
    def _validate_end_effector_frame(self):
        """Validate that the end effector frame exists."""
        try:
            frame_id = self.urdf_model.getFrameId(self.end_effector_frame)
            if frame_id >= self.urdf_model.nframes:
                raise ValueError(f"Frame {self.end_effector_frame} not found in URDF")
        except Exception:
            available_frames = [self.urdf_model.frames[i].name for i in range(self.urdf_model.nframes)]
            raise ValueError(f"Frame {self.end_effector_frame} not found in URDF. Available frames: {available_frames}")
            
    def _setup_tasks(self, initial_configuration: Optional[np.ndarray] = None):
        """Set up Pink tasks and configuration."""
        print("🔧 Setting up Pink tasks and limits...")
        
        # Create initial configuration
        if initial_configuration is None:
            q0 = pin.neutral(self.urdf_model)
        else:
            q0 = initial_configuration
                    
        self.configuration = pink.Configuration(self.urdf_model, self.urdf_model_data, q0)
        
        # Get current end effector transform
        current_transform = self.configuration.get_transform_frame_to_world(self.end_effector_frame)
        
        # Set up end effector task
        self.ee_task = FrameTask(
            self.end_effector_frame,
            position_cost=self.position_cost,
            orientation_cost=self.orientation_cost,
            lm_damping=self.lm_damping,
        )
        self.ee_task.set_target(current_transform)
        
        # Set up posture task
        self.posture_task = PostureTask(cost=self.posture_cost)
        self.posture_task.set_target(q0)
        
        # Set up damping task
        self.damping_task = DampingTask(cost=self.damping_cost)
        
        print("✅ Tasks configured!")
        
    def update_task_parameters(
        self,
        position_cost: Optional[float] = None,
        orientation_cost: Optional[float] = None,
        lm_damping: Optional[float] = None,
        posture_cost: Optional[float] = None,
        damping_cost: Optional[float] = None,
        integration_time_step: Optional[float] = None
    ):
        """Update task parameters dynamically."""
        if position_cost is not None:
            self.position_cost = position_cost
            self.ee_task.position_cost = position_cost
            
        if orientation_cost is not None:
            self.orientation_cost = orientation_cost
            self.ee_task.orientation_cost = orientation_cost
            
        if lm_damping is not None:
            self.lm_damping = lm_damping
            self.ee_task.lm_damping = lm_damping
            
        if posture_cost is not None:
            self.posture_cost = posture_cost
            self.posture_task.cost = posture_cost
            
        if damping_cost is not None:
            self.damping_cost = damping_cost
            self.damping_task.cost = damping_cost
            
        if integration_time_step is not None:
            self.integration_time_step = integration_time_step
            
    def set_target_transform(self, target_transform: pin.SE3):
        """Set the target transform for the end effector."""
        self.ee_task.set_target(target_transform)
        
    def set_target_pose(self, position: np.ndarray, orientation: np.ndarray):
        """
        Set target pose from position and orientation.
        
        Args:
            position: 3D position vector
            orientation: 3x3 rotation matrix or 4-element quaternion (wxyz)
        """
        if orientation.shape == (4,):
            # Quaternion (wxyz) to rotation matrix
            target_rotation = Rotation.from_quat([orientation[1], orientation[2], orientation[3], orientation[0]])
            rotation_matrix = target_rotation.as_matrix()
        elif orientation.shape == (3, 3):
            # Already a rotation matrix
            rotation_matrix = orientation
        else:
            raise ValueError("Orientation must be a 3x3 matrix or 4-element quaternion")
            
        target_transform = pin.SE3(rotation_matrix, position)
        self.set_target_transform(target_transform)
        
    def solve_ik(self, dt: Optional[float] = None) -> Tuple[np.ndarray, bool]:
        """
        Solve inverse kinematics for current target.
        
        Args:
            dt: Integration time step (uses instance default if None)
            
        Returns:
            Tuple of (joint_velocities, success_flag)
        """
        if dt is None:
            dt = self.integration_time_step
            
        start_time = time.time()
        
        try:
            # Prepare tasks and limits
            tasks = (self.ee_task, self.posture_task, self.damping_task)
            limits = (
                self.configuration.model.configuration_limit,
                self.configuration.model.velocity_limit,
                # self.acceleration_limit_obj,  # Commented out as in original
            )
            
            # Solve differential IK
            velocity = pink.solve_ik(
                self.configuration, tasks, dt, solver=self.solver_name, limits=limits
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
                
            return velocity, True
            
        except Exception as e:
            print(f"❌ IK solve failed: {e}")
            return np.zeros(self.urdf_model.nv), False
            
    def get_current_configuration(self) -> np.ndarray:
        """Get current joint configuration."""
        return self.configuration.q.copy()
        
    def get_current_end_effector_transform(self) -> pin.SE3:
        """Get current end effector transform."""
        return self.configuration.get_transform_frame_to_world(self.end_effector_frame)
        
    def get_current_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current end effector pose.
        
        Returns:
            Tuple of (position, rotation_matrix)
        """
        transform = self.get_current_end_effector_transform()
        return transform.translation.copy(), transform.rotation.copy()
        
    def get_joint_angles_string(self) -> str:
        """Get formatted string of current joint angles."""
        joint_angles_str = "Joint Angles (rad):\n"
        for i, angle in enumerate(self.configuration.q):
            joint_angles_str += f"  Joint {i+1}: {angle:.3f}\n"
        return joint_angles_str
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get solver statistics."""
        if not self.solve_times:
            return {"last_solve_time_ms": 0.0, "avg_solve_time_ms": 0.0, "max_solve_time_ms": 0.0}
            
        return {
            "last_solve_time_ms": self.last_solve_time,
            "avg_solve_time_ms": np.mean(self.solve_times),
            "max_solve_time_ms": np.max(self.solve_times),
            "solve_count": len(self.solve_times)
        }
        
    def set_configuration(self, joint_config: np.ndarray):
        """
        Set the robot to a specific joint configuration.
        
        Args:
            joint_config: Array of joint angles to set
        """
        if len(joint_config) != self.urdf_model.nq:
            raise ValueError(f"Joint configuration must have {self.urdf_model.nq} values, got {len(joint_config)}")
        
        self.configuration = pink.Configuration(self.urdf_model, self.urdf_model_data, joint_config)
        self.posture_task.set_target(joint_config)
        
        # Update end effector task target to current position
        current_transform = self.configuration.get_transform_frame_to_world(self.end_effector_frame)
        self.ee_task.set_target(current_transform)
    
    def reset_to_neutral(self):
        """Reset robot to neutral configuration."""
        q0 = pin.neutral(self.urdf_model)
        self.set_configuration(q0)
        
    def get_available_frames(self) -> list:
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
            "available_frames": self.get_available_frames()
        }
