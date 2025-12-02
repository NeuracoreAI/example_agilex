#!/usr/bin/env python3
"""Shared robot visualizer for Piper robot demos.

This module provides a clean interface for visualizing robot state using Viser,
encapsulating all the repeated setup, GUI controls, and update logic.
"""

from typing import Any

import numpy as np
import viser
import yourdfpy
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf


class RobotVisualizer:
    """Shared visualizer for robot demos.

    Encapsulates viser server setup, GUI controls, and update logic.
    """

    def __init__(self, urdf_path: str) -> None:
        """Initialize the visualizer.

        Args:
            urdf_path: Path to URDF file for robot visualization
        """
        # Initialize viser server
        self.server = viser.ViserServer()
        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        # Load URDF for visualization
        urdf = yourdfpy.URDF.load(urdf_path)
        self.urdf_vis = ViserUrdf(self.server, urdf, root_node_name="/robot")

        # GUI handles (initialized as None, created on demand) - all private
        self._timing_handle = None
        self._joint_angles_handle = None
        self._robot_status_handle = None
        self._safety_status_handle = None
        self._teleop_status_handle = None
        self._controller_status_handle = None
        self._gripper_status_handle = None
        self._homing_status_handle = None
        self._recording_status_handle = None

        # Pink parameter handles
        self._position_weight_handle = None
        self._orientation_weight_handle = None
        self._frame_task_gain_handle = None
        self._lm_damping_handle = None
        self._damping_weight_handle = None
        self._solver_damping_value_handle = None
        self._posture_cost_handles: list[Any] = []

        # Robot control handles
        self._enable_robot_handle = None
        self._disable_robot_handle = None
        self._emergency_stop_handle = None
        self._go_home_button = None

        # Teleop-specific handles
        self._grip_value_handle = None
        self._trigger_value_handle = None

        # Rate control handles
        self._ik_solver_rate_handle = None
        self._controller_min_cutoff_handle = None
        self._controller_beta_handle = None
        self._controller_d_cutoff_handle = None

        self._grip_threshold_handle = None
        self._translation_scale_handle = None
        self._rotation_scale_handle = None

        # Recording handles
        self._start_recording_button = None
        self._stop_recording_button = None

        # Visualization handles
        self._controller_handle = None
        self._target_frame_handle = None
        self._ik_target_handle = None

        # Internal state
        self._ema_timing = 0.001

    def add_basic_controls(self) -> None:
        """Add basic GUI controls (timing, joint angles)."""
        self._timing_handle = self.server.gui.add_number(
            "IK Solve Time (ms)", 0.001, disabled=True
        )
        self._joint_angles_handle = self.server.gui.add_text(
            "Joint Angles", "Waiting for IK solution..."
        )

    def add_robot_status_controls(self) -> None:
        """Add robot status display controls."""
        self._robot_status_handle = self.server.gui.add_text(
            "Robot Status", "Initializing..."
        )

    def add_safety_status_controls(self) -> None:
        """Add safety status display controls."""
        self._safety_status_handle = self.server.gui.add_text(
            "Safety Status", "Monitoring..."
        )

    def add_teleop_controls(self) -> None:
        """Add teleoperation-specific controls."""
        self._grip_value_handle = self.server.gui.add_number(
            "Grip Value", 0.0, disabled=True
        )
        self._trigger_value_handle = self.server.gui.add_number(
            "Trigger Value (Gripper)", 0.0, disabled=True
        )
        self._teleop_status_handle = self.server.gui.add_text(
            "Teleop Status", "Inactive"
        )
        self._controller_status_handle = self.server.gui.add_text(
            "Controller Status", "Waiting..."
        )

    def add_gripper_status_controls(self) -> None:
        """Add gripper status display controls."""
        self._gripper_status_handle = self.server.gui.add_text(
            "Gripper Status", "Open (0%)"
        )

    def add_homing_controls(self) -> None:
        """Add homing controls."""
        self._homing_status_handle = self.server.gui.add_text("Homing Status", "Idle")
        self._go_home_button = self.server.gui.add_button("Go Home")

    def add_robot_control_buttons(self) -> None:
        """Add robot control buttons (enable, disable, emergency stop).

        Note: For homing functionality, use add_homing_controls() instead.
        """
        self._enable_robot_handle = self.server.gui.add_button("Enable Robot")
        self._disable_robot_handle = self.server.gui.add_button("Disable Robot")
        self._emergency_stop_handle = self.server.gui.add_button("Emergency Stop")

    def add_pink_parameter_controls(
        self,
        position_cost: float,
        orientation_cost: float,
        frame_task_gain: float,
        lm_damping: float,
        damping_cost: float,
        solver_damping_value: float,
        posture_cost_vector: list[float],
    ) -> None:
        """Add Pink IK parameter controls.

        Args:
            position_cost: Initial position cost value
            orientation_cost: Initial orientation cost value
            frame_task_gain: Initial frame task gain value
            lm_damping: Initial LM damping value
            damping_cost: Initial damping cost value
            solver_damping_value: Initial solver damping value
            posture_cost_vector: Initial posture cost vector (one value per joint)
        """
        self._position_weight_handle = self.server.gui.add_number(
            "Position Weight", position_cost, min=0.0, max=10.0, step=0.1
        )
        self._orientation_weight_handle = self.server.gui.add_number(
            "Orientation Weight", orientation_cost, min=0.0, max=1.0, step=0.01
        )
        self._frame_task_gain_handle = self.server.gui.add_number(
            "Frame Task Gain", frame_task_gain, min=0.0, max=10.0, step=0.1
        )
        self._lm_damping_handle = self.server.gui.add_number(
            "LM Damping", lm_damping, min=0.0, max=5.0, step=0.01
        )
        self._damping_weight_handle = self.server.gui.add_number(
            "Damping Weight", damping_cost, min=0.0, max=1.0, step=0.01
        )
        self._solver_damping_value_handle = self.server.gui.add_number(
            "Solver Damping Value", solver_damping_value, min=0.0, max=1.0, step=0.0001
        )

        # Posture cost controls (one per joint)
        self._posture_cost_handles = []
        for i in range(len(posture_cost_vector)):
            handle = self.server.gui.add_number(
                f"Posture Cost J{i+1}",
                posture_cost_vector[i],
                min=0.0,
                max=1.0,
                step=0.01,
            )
            self._posture_cost_handles.append(handle)

    def add_controller_filter_controls(
        self,
        initial_min_cutoff: float,
        initial_beta: float,
        initial_d_cutoff: float,
    ) -> None:
        """Add 1€ Filter parameter controls for controller.

        Args:
            initial_min_cutoff: Initial minimum cutoff frequency
            initial_beta: Initial speed coefficient
            initial_d_cutoff: Initial derivative cutoff frequency
        """
        self._controller_min_cutoff_handle = self.server.gui.add_number(
            "Controller Min Cutoff",
            initial_min_cutoff,
            min=0.01,
            max=10.0,
            step=0.01,
        )
        self._controller_beta_handle = self.server.gui.add_number(
            "Controller Beta", initial_beta, min=0.0, max=10.0, step=0.01
        )
        self._controller_d_cutoff_handle = self.server.gui.add_number(
            "Controller D Cutoff",
            initial_d_cutoff,
            min=0.01,
            max=10.0,
            step=0.01,
        )

    def add_grip_threshold_control(self, initial_threshold: float) -> None:
        """Add grip threshold control.

        Args:
            initial_threshold: Initial grip threshold value (0.0-1.0)
        """
        self._grip_threshold_handle = self.server.gui.add_number(
            "Grip Threshold", initial_threshold, min=0.0, max=1.0, step=0.05
        )

    def add_scaling_controls(
        self, initial_translation_scale: float, initial_rotation_scale: float
    ) -> None:
        """Add scaling factor controls for translation and rotation.

        Args:
            initial_translation_scale: Initial translation scale factor
            initial_rotation_scale: Initial rotation scale factor
        """
        self._translation_scale_handle = self.server.gui.add_number(
            "Translation Scale",
            initial_translation_scale,
            min=0.1,
            max=10.0,
            step=0.001,
        )
        self._rotation_scale_handle = self.server.gui.add_number(
            "Rotation Scale", initial_rotation_scale, min=0.1, max=10.0, step=0.001
        )

    def add_recording_controls(self, record: bool = False) -> None:
        """Add recording controls for neuracore."""
        self._recording_status_handle = self.server.gui.add_text(
            "Recording Status", "Not Recording" if not record else "Ready to Record"
        )
        if record:
            self._start_recording_button = self.server.gui.add_button("Start Recording")
            self._stop_recording_button = self.server.gui.add_button("Stop Recording")

    def add_ik_target_controls(
        self, initial_position: np.ndarray, initial_orientation: np.ndarray
    ) -> None:
        """Add interactive IK target controls.

        Args:
            initial_position: Initial target position (3D array)
            initial_orientation: Initial target orientation (3x3 rotation matrix)
        """
        quat_xyzw = Rotation.from_matrix(initial_orientation).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        self._ik_target_handle = self.server.scene.add_transform_controls(
            "/ik_target",
            scale=0.2,
            position=tuple(initial_position),
            wxyz=tuple(quat_wxyz),
        )

    def add_controller_visualization(self) -> None:
        """Add controller transform visualization."""
        self._controller_handle = self.server.scene.add_transform_controls(
            "/controller",
            scale=0.15,
            position=(0, 0, 0),
            wxyz=(1, 0, 0, 0),
        )

    def add_target_frame_visualization(self) -> None:
        """Add target/goal frame visualization."""
        self._target_frame_handle = self.server.scene.add_frame(
            "/target_goal", axes_length=0.1, axes_radius=0.003
        )

    def update_robot_pose(self, joint_config: np.ndarray) -> None:
        """Update robot visualization from joint configuration.

        Args:
            joint_config: Joint angles in radians
        """
        self.urdf_vis.update_cfg(joint_config)

    def update_joint_angles_display(
        self, joint_config: np.ndarray, show_gripper: bool = False
    ) -> None:
        """Update joint angles display.

        Args:
            joint_config: Joint angles in radians
            show_gripper: Whether to show gripper joints (joints 7&8)
        """
        if self._joint_angles_handle is None:
            return

        joint_angles_str = "Joint Angles (rad):\n"
        joint_angles_deg = np.degrees(joint_config)
        num_joints = len(joint_config) if show_gripper else 6

        for i in range(num_joints):
            angle_rad = joint_config[i]
            angle_deg = joint_angles_deg[i]
            joint_type = "Robot" if i < 6 else "Gripper"
            label = f"Joint {i+1} ({joint_type})" if show_gripper else f"J{i+1}"
            joint_angles_str += f"  {label}: {angle_rad:.3f} rad ({angle_deg:.1f}°)\n"

        self._joint_angles_handle.value = joint_angles_str

    def update_timing(self, solve_time_ms: float) -> None:
        """Update timing display with exponential moving average.

        Args:
            solve_time_ms: IK solve time in milliseconds
        """
        if self._timing_handle is None:
            return

        self._ema_timing = 0.99 * self._ema_timing + 0.01 * solve_time_ms
        self._timing_handle.value = self._ema_timing

    def update_robot_status(self, status: str) -> None:
        """Update robot status display.

        Args:
            status: Status string to display
        """
        if self._robot_status_handle is not None:
            self._robot_status_handle.value = status

    def update_safety_status(self, status: str) -> None:
        """Update safety status display.

        Args:
            status: Status string to display
        """
        if self._safety_status_handle is not None:
            self._safety_status_handle.value = status

    def update_teleop_status(self, active: bool) -> None:
        """Update teleop status display.

        Args:
            active: Whether teleop is active
        """
        if self._teleop_status_handle is not None:
            self._teleop_status_handle.value = (
                "Teleop Status: Active" if active else "Teleop Status: Inactive"
            )

    def update_controller_status_display(
        self, position: np.ndarray | None, connected: bool = True
    ) -> None:
        """Update controller status display.

        Args:
            position: Controller position (3D array) or None
            connected: Whether controller is connected
        """
        if self._controller_status_handle is None:
            return

        if connected and position is not None:
            controller_status_str = "Controller Status:\n"
            controller_status_str += f"  Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]\n"
            controller_status_str += "  Connected: ✓\n"
            self._controller_status_handle.value = controller_status_str
        else:
            self._controller_status_handle.value = "Controller Status:\n  Connected: ✗"

    def update_gripper_status(
        self, trigger_value: float, robot_enabled: bool = True
    ) -> None:
        """Update gripper status display.

        Args:
            trigger_value: Trigger value (0.0 = open, 1.0 = closed)
            robot_enabled: Whether robot is enabled
        """
        if self._gripper_status_handle is None:
            return

        gripper_closed_percent = trigger_value * 100.0
        if trigger_value > 0.9:
            gripper_state = "Closed"
        elif trigger_value > 0.1:
            gripper_state = "Closing"
        else:
            gripper_state = "Open"

        status = f"Gripper: {gripper_state} ({gripper_closed_percent:.0f}% closed)"
        if not robot_enabled:
            status += " [Disabled]"

        self._gripper_status_handle.value = status

    def update_homing_status(self, status: str) -> None:
        """Update homing status display.

        Args:
            status: Homing status string
        """
        if self._homing_status_handle is not None:
            self._homing_status_handle.value = status

    def update_recording_status(self, status: str) -> None:
        """Update recording status display.

        Args:
            status: Recording status string
        """
        if self._recording_status_handle is not None:
            self._recording_status_handle.value = status

    def update_controller_visualization(self, transform: np.ndarray | None) -> None:
        """Update controller transform visualization.

        Args:
            transform: 4x4 transformation matrix or None
        """
        if self._controller_handle is None or transform is None:
            return

        controller_pos = transform[:3, 3]
        controller_rot = Rotation.from_matrix(transform[:3, :3])
        controller_quat_xyzw = controller_rot.as_quat()
        controller_quat_wxyz = [
            controller_quat_xyzw[3],
            controller_quat_xyzw[0],
            controller_quat_xyzw[1],
            controller_quat_xyzw[2],
        ]

        self._controller_handle.position = tuple(controller_pos)
        self._controller_handle.wxyz = tuple(controller_quat_wxyz)

    def update_target_visualization(self, transform: np.ndarray | None) -> None:
        """Update target/goal frame visualization.

        Args:
            transform: 4x4 transformation matrix or None
        """
        if self._target_frame_handle is None or transform is None:
            return

        target_pos = transform[:3, 3]
        target_rot = Rotation.from_matrix(transform[:3, :3])
        target_quat_xyzw = target_rot.as_quat()
        target_quat_wxyz = [
            target_quat_xyzw[3],
            target_quat_xyzw[0],
            target_quat_xyzw[1],
            target_quat_xyzw[2],
        ]

        self._target_frame_handle.position = tuple(target_pos)
        self._target_frame_handle.wxyz = tuple(target_quat_wxyz)

    def get_ik_target_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get IK target pose from GUI controls.

        Returns:
            Tuple of (position, rotation_matrix)

        Raises:
            ValueError: If IK target controls not initialized
        """
        if self._ik_target_handle is None:
            raise ValueError("IK target controls not initialized")

        target_position = np.array(self._ik_target_handle.position)
        target_wxyz = np.array(self._ik_target_handle.wxyz)

        # Convert wxyz quaternion to rotation matrix
        target_rotation = Rotation.from_quat(
            [target_wxyz[1], target_wxyz[2], target_wxyz[3], target_wxyz[0]]
        )  # wxyz to xyzw
        target_rotation_matrix = target_rotation.as_matrix()

        return target_position, target_rotation_matrix

    def set_ik_target_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        """Set IK target pose in GUI controls.

        Args:
            position: Target position (3D array)
            orientation: Target orientation (3x3 rotation matrix)

        Raises:
            ValueError: If IK target controls not initialized
        """
        if self._ik_target_handle is None:
            raise ValueError("IK target controls not initialized")

        quat_xyzw = Rotation.from_matrix(orientation).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        self._ik_target_handle.position = tuple(position)
        self._ik_target_handle.wxyz = tuple(quat_wxyz)

    def get_pink_parameters(self) -> dict:
        """Get Pink IK parameters from GUI controls.

        Returns:
            Dictionary with parameter values

        Raises:
            ValueError: If Pink parameter controls not initialized
        """
        if not self._posture_cost_handles:
            raise ValueError("Pink parameter controls not initialized")

        if (
            self._position_weight_handle is None
            or self._orientation_weight_handle is None
            or self._frame_task_gain_handle is None
            or self._lm_damping_handle is None
            or self._damping_weight_handle is None
            or self._solver_damping_value_handle is None
        ):
            raise ValueError("Pink parameter controls not initialized")

        posture_cost_vector = np.array(
            [handle.value for handle in self._posture_cost_handles]
        )

        params = {
            "position_cost": self._position_weight_handle.value,
            "orientation_cost": self._orientation_weight_handle.value,
            "frame_task_gain": self._frame_task_gain_handle.value,
            "lm_damping": self._lm_damping_handle.value,
            "damping_cost": self._damping_weight_handle.value,
            "solver_damping_value": self._solver_damping_value_handle.value,
            "posture_cost_vector": posture_cost_vector,
        }
        return params

    def get_controller_filter_params(self) -> tuple[float, float, float]:
        """Get 1€ Filter parameters from GUI.

        Returns:
            Tuple of (min_cutoff, beta, d_cutoff)

        Raises:
            ValueError: If controller filter controls not initialized
        """
        if (
            self._controller_min_cutoff_handle is None
            or self._controller_beta_handle is None
            or self._controller_d_cutoff_handle is None
        ):
            raise ValueError("Controller filter controls not initialized")
        return (
            self._controller_min_cutoff_handle.value,
            self._controller_beta_handle.value,
            self._controller_d_cutoff_handle.value,
        )

    def get_translation_scale(self) -> float:
        """Get translation scale value from GUI.

        Returns:
            Translation scale value

        Raises:
            ValueError: If scaling controls not initialized
        """
        if self._translation_scale_handle is None:
            raise ValueError("Scaling controls not initialized")
        return self._translation_scale_handle.value

    def get_rotation_scale(self) -> float:
        """Get rotation scale value from GUI.

        Returns:
            Rotation scale value

        Raises:
            ValueError: If scaling controls not initialized
        """
        if self._rotation_scale_handle is None:
            raise ValueError("Scaling controls not initialized")
        return self._rotation_scale_handle.value

    def get_grip_threshold(self) -> float:
        """Get grip threshold value from GUI.

        Returns:
            Grip threshold value

        Raises:
            ValueError: If grip threshold control has not been initialized
        """
        if self._grip_threshold_handle is None:
            raise ValueError("Grip threshold control not initialized")
        return self._grip_threshold_handle.value

    def add_ik_solver_rate_control(self, initial_rate: float) -> None:
        """Add IK solver rate control to GUI.

        Args:
            initial_rate: Initial IK solver rate in Hz
        """
        self._ik_solver_rate_handle = self.server.gui.add_number(
            "IK Solver Rate (Hz)",
            initial_rate,
            min=1.0,
            max=1000.0,
            step=1.0,
        )

    def get_ik_solver_rate(self) -> float:
        """Get IK solver rate from GUI.

        Returns:
            IK solver rate in Hz

        Raises:
            ValueError: If IK solver rate control has not been initialized
        """
        if self._ik_solver_rate_handle is None:
            raise ValueError("IK solver rate control not initialized")
        return self._ik_solver_rate_handle.value

    def set_grip_value(self, value: float) -> None:
        """Set grip value display.

        Args:
            value: Grip value (0.0-1.0)

        Raises:
            ValueError: If grip value control not initialized
        """
        if self._grip_value_handle is None:
            raise ValueError("Grip value control not initialized")
        self._grip_value_handle.value = value

    def set_trigger_value(self, value: float) -> None:
        """Set trigger value display.

        Args:
            value: Trigger value (0.0-1.0)

        Raises:
            ValueError: If trigger value control not initialized
        """
        if self._trigger_value_handle is None:
            raise ValueError("Trigger value control not initialized")
        self._trigger_value_handle.value = value

    def set_joint_angles_text(self, text: str) -> None:
        """Set joint angles text display.

        Args:
            text: Text to display

        Raises:
            ValueError: If joint angles control not initialized
        """
        if self._joint_angles_handle is None:
            raise ValueError("Joint angles control not initialized")
        self._joint_angles_handle.value = text

    # Button state getters and setters
    def is_enable_robot_pressed(self) -> bool:
        """Check if enable robot button was pressed.

        Returns:
            True if button was pressed, False otherwise

        Raises:
            ValueError: If enable robot button not initialized
        """
        if self._enable_robot_handle is None:
            raise ValueError("Enable robot button not initialized")
        return self._enable_robot_handle.value

    def reset_enable_robot_button(self) -> None:
        """Reset enable robot button state.

        Raises:
            ValueError: If enable robot button not initialized
        """
        if self._enable_robot_handle is None:
            raise ValueError("Enable robot button not initialized")
        self._enable_robot_handle.value = False

    def is_disable_robot_pressed(self) -> bool:
        """Check if disable robot button was pressed.

        Returns:
            True if button was pressed, False otherwise

        Raises:
            ValueError: If disable robot button not initialized
        """
        if self._disable_robot_handle is None:
            raise ValueError("Disable robot button not initialized")
        return self._disable_robot_handle.value

    def reset_disable_robot_button(self) -> None:
        """Reset disable robot button state.

        Raises:
            ValueError: If disable robot button not initialized
        """
        if self._disable_robot_handle is None:
            raise ValueError("Disable robot button not initialized")
        self._disable_robot_handle.value = False

    def is_emergency_stop_pressed(self) -> bool:
        """Check if emergency stop button was pressed.

        Returns:
            True if button was pressed, False otherwise

        Raises:
            ValueError: If emergency stop button not initialized
        """
        if self._emergency_stop_handle is None:
            raise ValueError("Emergency stop button not initialized")
        return self._emergency_stop_handle.value

    def reset_emergency_stop_button(self) -> None:
        """Reset emergency stop button state.

        Raises:
            ValueError: If emergency stop button not initialized
        """
        if self._emergency_stop_handle is None:
            raise ValueError("Emergency stop button not initialized")
        self._emergency_stop_handle.value = False

    def is_go_home_pressed(self) -> bool:
        """Check if go home button was pressed.

        Returns:
            True if button was pressed, False otherwise

        Raises:
            ValueError: If go home button not initialized
        """
        if self._go_home_button is None:
            raise ValueError("Go home button not initialized")
        return self._go_home_button.value

    def reset_go_home_button(self) -> None:
        """Reset go home button state.

        Raises:
            ValueError: If go home button not initialized
        """
        if self._go_home_button is None:
            raise ValueError("Go home button not initialized")
        self._go_home_button.value = False

    def is_start_recording_pressed(self) -> bool:
        """Check if start recording button was pressed.

        Returns:
            True if button was pressed, False otherwise

        Raises:
            ValueError: If start recording button not initialized
        """
        if self._start_recording_button is None:
            raise ValueError("Start recording button not initialized")
        return self._start_recording_button.value

    def reset_start_recording_button(self) -> None:
        """Reset start recording button state.

        Raises:
            ValueError: If start recording button not initialized
        """
        if self._start_recording_button is None:
            raise ValueError("Start recording button not initialized")
        self._start_recording_button.value = False

    def is_stop_recording_pressed(self) -> bool:
        """Check if stop recording button was pressed.

        Returns:
            True if button was pressed, False otherwise

        Raises:
            ValueError: If stop recording button not initialized
        """
        if self._stop_recording_button is None:
            raise ValueError("Stop recording button not initialized")
        return self._stop_recording_button.value

    def reset_stop_recording_button(self) -> None:
        """Reset stop recording button state.

        Raises:
            ValueError: If stop recording button not initialized
        """
        if self._stop_recording_button is None:
            raise ValueError("Stop recording button not initialized")
        self._stop_recording_button.value = False

    def stop(self) -> None:
        """Stop the visualizer server."""
        self.server.stop()
