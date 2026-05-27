from typing import Any, Callable
import numpy as np
import viser

class RobotVisualizerGUI:
    """Handles all the 2D UI elements: buttons, sliders, and text readouts."""
    
    def __init__(self, server: viser.ViserServer) -> None:
        self.server = server
        self._ema_timing = 0.001
        
        # State Handles
        self._timing_handle = None
        self._joint_angles_handle = None
        self._robot_status_handle = None
        self._teleop_status_handle = None
        self._controller_status_handle = None
        self._gripper_status_handle = None
        self._policy_status_handle = None
        
        # Input Handles
        self._position_weight_handle = None
        self._orientation_weight_handle = None
        self._frame_task_gain_handle = None
        self._lm_damping_handle = None
        self._damping_weight_handle = None
        self._solver_damping_value_handle = None
        self._posture_cost_handles = []
        
        self._controller_min_cutoff_handle = None
        self._controller_beta_handle = None
        self._controller_d_cutoff_handle = None
        self._translation_scale_handle = None
        self._rotation_scale_handle = None
        
        self._prediction_ratio_handle = None
        self._policy_execution_rate_handle = None
        self._robot_rate_handle = None
        self._execution_mode_dropdown = None
        
        self._grip_value_handle = None
        self._trigger_value_handle = None
        
        # Button Handles
        self._enable_robot_handle = None
        self._disable_robot_handle = None
        self._emergency_stop_handle = None
        self._go_home_button = None
        self._toggle_robot_enabled_status_button = None
        self._run_policy_button = None
        self._start_policy_execution_button = None
        self._play_policy_button = None

        self._slow_translation_scale_handle = None
        self._slow_rotation_scale_handle = None
        self._wrist_step_handle = None
        self._save_config_button = None

    def add_advanced_tuning_controls(self, initial_slow_t: float, initial_slow_r: float, initial_wrist_step: float) -> None:
        """Adds advanced sliders for precision movement and button toggles."""
        self._slow_translation_scale_handle = self.server.gui.add_number("Slow Translation Scale", initial_slow_t, min=0.1, max=5.0, step=0.01)
        self._slow_rotation_scale_handle = self.server.gui.add_number("Slow Rotation Scale", initial_slow_r, min=0.1, max=5.0, step=0.01)
        self._wrist_step_handle = self.server.gui.add_number("Wrist Nudge (Degrees)", initial_wrist_step, min=1.0, max=45.0, step=1.0)
        self._save_config_button = self.server.gui.add_button("💾 Save Config to YAML")

    def get_slow_translation_scale(self) -> float:
        return self._slow_translation_scale_handle.value

    def get_slow_rotation_scale(self) -> float:
        return self._slow_rotation_scale_handle.value

    def get_wrist_step_degrees(self) -> float:
        return self._wrist_step_handle.value

    def set_save_config_callback(self, cb: Callable[[], Any]) -> None:
        if self._save_config_button: 
            self._save_config_button.on_click(lambda _: cb())

    def add_basic_controls(self) -> None:
        self._timing_handle = self.server.gui.add_number("IK Solve Time (ms)", 0.001, disabled=True)
        self._joint_angles_handle = self.server.gui.add_text("Joint Angles", "Waiting...")

    def add_robot_status_controls(self) -> None:
        self._robot_status_handle = self.server.gui.add_text("Robot Status", "Initializing...")

    def add_teleop_controls(self) -> None:
        self._grip_value_handle = self.server.gui.add_number("Grip Value", 0.0, disabled=True)
        self._trigger_value_handle = self.server.gui.add_number("Trigger", 0.0, disabled=True)
        self._teleop_status_handle = self.server.gui.add_text("Teleop Status", "Inactive")
        self._controller_status_handle = self.server.gui.add_text("Controller Status", "Waiting...")

    def add_gripper_status_controls(self) -> None:
        self._gripper_status_handle = self.server.gui.add_text("Gripper Status", "Open (0%)")

    def add_homing_controls(self) -> None:
        self._go_home_button = self.server.gui.add_button("Go Home")

    def add_toggle_robot_enabled_status_button(self) -> None:
        self._toggle_robot_enabled_status_button = self.server.gui.add_button("Enable Robot")

    def add_controller_filter_controls(self, initial_min_cutoff: float, initial_beta: float, initial_d_cutoff: float) -> None:
        self._controller_min_cutoff_handle = self.server.gui.add_number("Controller Min Cutoff", initial_min_cutoff, min=0.01, max=10.0, step=0.01)
        self._controller_beta_handle = self.server.gui.add_number("Controller Beta", initial_beta, min=0.0, max=10.0, step=0.01)
        self._controller_d_cutoff_handle = self.server.gui.add_number("Controller D Cutoff", initial_d_cutoff, min=0.01, max=10.0, step=0.01)

    def add_scaling_controls(self, initial_translation_scale: float, initial_rotation_scale: float) -> None:
        self._translation_scale_handle = self.server.gui.add_number("Translation Scale", initial_translation_scale, min=0.1, max=10.0, step=0.001)
        self._rotation_scale_handle = self.server.gui.add_number("Rotation Scale", initial_rotation_scale, min=0.1, max=10.0, step=0.001)

    def add_pink_parameter_controls(self, position_cost: float, orientation_cost: float, frame_task_gain: float, lm_damping: float, damping_cost: float, solver_damping_value: float, posture_cost_vector: list[float]) -> None:
        self._position_weight_handle = self.server.gui.add_number("Position Weight", position_cost, min=0.0, max=10.0, step=0.1)
        self._orientation_weight_handle = self.server.gui.add_number("Orientation Weight", orientation_cost, min=0.0, max=1.0, step=0.01)
        self._frame_task_gain_handle = self.server.gui.add_number("Frame Task Gain", frame_task_gain, min=0.0, max=10.0, step=0.1)
        self._lm_damping_handle = self.server.gui.add_number("LM Damping", lm_damping, min=0.0, max=5.0, step=0.01)
        self._damping_weight_handle = self.server.gui.add_number("Damping Weight", damping_cost, min=0.0, max=1.0, step=0.01)
        self._solver_damping_value_handle = self.server.gui.add_number("Solver Damping Value", solver_damping_value, min=0.0, max=1.0, step=0.0001)
        
        for i, cost in enumerate(posture_cost_vector):
            self._posture_cost_handles.append(self.server.gui.add_number(f"Posture Cost J{i+1}", cost, min=0.0, max=1.0, step=0.01))

    def add_policy_controls(self, initial_prediction_ratio: float = 0.8, initial_policy_rate: float = 200.0, initial_robot_rate: float = 200.0, initial_execution_mode: str = "targeting_time") -> None:
        self._policy_status_handle = self.server.gui.add_text("Policy Status", "Ready")
        self._prediction_ratio_handle = self.server.gui.add_number("Prediction Ratio", initial_prediction_ratio, min=0.0, max=1.0, step=0.01)
        self._policy_execution_rate_handle = self.server.gui.add_number("Policy Rate", initial_policy_rate, min=1.0, max=200.0, step=1.0)
        self._robot_rate_handle = self.server.gui.add_number("Robot Rate", initial_robot_rate, min=1.0, max=200.0, step=1.0)
        self._execution_mode_dropdown = self.server.gui.add_dropdown("Execution Mode", options=["targeting_time", "targeting_pose"], initial_value=initial_execution_mode)

    def add_policy_buttons(self) -> None:
        self._run_policy_button = self.server.gui.add_button("Run Policy (Preview)")
        self._start_policy_execution_button = self.server.gui.add_button("Execute Policy (Run Preview)")
        self._play_policy_button = self.server.gui.add_button("Continuous Receding Horizon")

    # --- UI Update Methods ---
    def update_timing(self, solve_time_ms: float) -> None:
        if self._timing_handle:
            self._ema_timing = 0.99 * self._ema_timing + 0.01 * solve_time_ms
            self._timing_handle.value = self._ema_timing

    def update_robot_status(self, status: str) -> None:
        if self._robot_status_handle: self._robot_status_handle.value = status

    def update_teleop_status(self, active: bool) -> None:
        if self._teleop_status_handle: self._teleop_status_handle.value = f"Teleop Status: {'Active' if active else 'Inactive'}"

    def update_controller_status_display(self, position: np.ndarray | None, connected: bool = True) -> None:
        if not self._controller_status_handle: return
        if connected and position is not None:
            self._controller_status_handle.value = f"Controller Status:\n  Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]\n  Connected: ✓"
        else:
            self._controller_status_handle.value = "Controller Status:\n  Connected: ✗"

    def update_gripper_status(self, trigger_value: float, robot_enabled: bool = True) -> None:
        if not self._gripper_status_handle: return
        state = "Closed" if trigger_value > 0.9 else "Closing" if trigger_value > 0.1 else "Open"
        status = f"Gripper: {state} ({trigger_value * 100.0:.0f}% closed)"
        self._gripper_status_handle.value = status + ("" if robot_enabled else " [Disabled]")

    def update_policy_status(self, status: str) -> None:
        if self._policy_status_handle: self._policy_status_handle.value = status

    def update_toggle_robot_enabled_status(self, enabled: bool) -> None:
        if self._toggle_robot_enabled_status_button:
            self._toggle_robot_enabled_status_button.label = "Disable Robot" if enabled else "Enable Robot"

    def update_play_policy_button_status(self, active: bool) -> None:
        if self._play_policy_button:
            self._play_policy_button.label = "Stop Continuous Horizon" if active else "Continuous Receding Horizon"

    def update_joint_angles_display(self, joint_config: np.ndarray, show_gripper: bool = False) -> None:
        if not self._joint_angles_handle: return
        lines = ["Joint Angles (rad):"]
        num_joints = len(joint_config) if show_gripper else 6
        for i in range(num_joints):
            lbl = f"Joint {i+1} ({'Robot' if i < 6 else 'Gripper'})" if show_gripper else f"J{i+1}"
            lines.append(f"  {lbl}: {joint_config[i]:.3f} rad ({np.degrees(joint_config[i]):.1f}°)")
        self._joint_angles_handle.value = "\n".join(lines)

    def set_grip_value(self, value: float) -> None:
        if self._grip_value_handle: self._grip_value_handle.value = value

    def set_trigger_value(self, value: float) -> None:
        if self._trigger_value_handle: self._trigger_value_handle.value = value

    # --- Getters ---
    def get_controller_filter_params(self) -> tuple[float, float, float]:
        return (self._controller_min_cutoff_handle.value, self._controller_beta_handle.value, self._controller_d_cutoff_handle.value)

    def get_translation_scale(self) -> float: return self._translation_scale_handle.value
    def get_rotation_scale(self) -> float: return self._rotation_scale_handle.value
    def get_prediction_ratio(self) -> float: return self._prediction_ratio_handle.value
    def get_policy_execution_rate(self) -> float: return self._policy_execution_rate_handle.value
    def get_robot_rate(self) -> float: return self._robot_rate_handle.value
    def get_execution_mode(self) -> str: return self._execution_mode_dropdown.value

    def get_pink_parameters(self) -> dict:
        return {
            "position_cost": self._position_weight_handle.value,
            "orientation_cost": self._orientation_weight_handle.value,
            "frame_task_gain": self._frame_task_gain_handle.value,
            "lm_damping": self._lm_damping_handle.value,
            "damping_cost": self._damping_weight_handle.value,
            "solver_damping_value": self._solver_damping_value_handle.value,
            "posture_cost_vector": np.array([h.value for h in self._posture_cost_handles]),
        }

    # --- Button Setters/Callbacks ---
    def set_toggle_robot_enabled_status_callback(self, cb: Callable[[], Any]) -> None:
        if self._toggle_robot_enabled_status_button: self._toggle_robot_enabled_status_button.on_click(lambda _: cb())
    def set_go_home_callback(self, cb: Callable[[], Any]) -> None:
        if self._go_home_button: self._go_home_button.on_click(lambda _: cb())
    def set_run_policy_callback(self, cb: Callable[[], Any]) -> None:
        if self._run_policy_button: self._run_policy_button.on_click(lambda _: cb())
    def set_start_policy_execution_callback(self, cb: Callable[[], Any]) -> None:
        if self._start_policy_execution_button: self._start_policy_execution_button.on_click(lambda _: cb())
    def set_play_policy_callback(self, cb: Callable[[], Any]) -> None:
        if self._play_policy_button: self._play_policy_button.on_click(lambda _: cb())
    def set_execution_mode_callback(self, cb: Callable[[], Any]) -> None:
        if self._execution_mode_dropdown: self._execution_mode_dropdown.on_update(lambda _: cb())

    def set_run_policy_button_disabled(self, disabled: bool) -> None:
        if self._run_policy_button: self._run_policy_button.disabled = disabled
    def set_start_policy_execution_button_disabled(self, disabled: bool) -> None:
        if self._start_policy_execution_button: self._start_policy_execution_button.disabled = disabled
    def set_play_policy_button_disabled(self, disabled: bool) -> None:
        if self._play_policy_button: self._play_policy_button.disabled = disabled