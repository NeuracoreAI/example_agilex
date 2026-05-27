"""Extracted actions for running and executing policies across different rollout scripts."""

import threading
import time
from typing import Any, Optional

import numpy as np
from common.configs import (
    CAMERA_NAMES,
    GRIPPER_NAME,
    JOINT_NAMES,
    MAX_ACTION_ERROR_THRESHOLD,
    MAX_SAFETY_THRESHOLD,
    POLICY_EXECUTION_RATE,
    TARGETING_POSE_TIME_THRESHOLD,
)
from common.data_manager import DataManager, RobotActivityState
from common.policy_helpers import (
    convert_predictions_to_horizon,
    log_robot_state_for_policy,
)
from common.policy_state import PolicyState


def run_policy(
    data_manager: DataManager,
    policy: Any,
    policy_state: PolicyState,
    visualizer: Optional[Any],
    input_embodiment_description: dict,
) -> bool:
    """Handle Run Policy trigger to capture state and get policy prediction."""
    print("Running policy...")

    # Use our helper to log only what the model expects
    if not log_robot_state_for_policy(data_manager, input_embodiment_description):
        print("✗ No data available to run policy")
        return False

    try:
        start_time = time.time()
        predictions = policy.predict(timeout=60)
        inference_time = time.time() - start_time
        prediction_horizon = convert_predictions_to_horizon(predictions)

        horizon_length = (
            len(next(iter(prediction_horizon.values()))) if prediction_horizon else 0
        )
        print(
            f"  ✓ Got policy prediction in {inference_time:.2f}s with horizon length {horizon_length}"
        )

        if visualizer:
            policy_state.set_execution_ratio(visualizer.get_prediction_ratio())

        policy_state.set_prediction_horizon(prediction_horizon)

        if visualizer:
            visualizer.update_ghost_robot_visibility(True)

        policy_state.set_ghost_robot_playing(True)
        policy_state.reset_ghost_action_index()
        return True
    except Exception as e:
        print(f"✗ Failed to get policy prediction: {e}")
        return False


def start_policy_execution(
    data_manager: DataManager, policy_state: PolicyState
) -> bool:
    """Handle Execute Policy trigger to start policy execution."""
    print("Starting policy execution...")
    state = data_manager.get_robot_activity_state()

    if (
        state == RobotActivityState.POLICY_CONTROLLED
        and not policy_state.get_continuous_play_active()
    ):
        print("⚠️ Policy execution already in progress")
        return False
    if state == RobotActivityState.DISABLED:
        print("⚠️ Cannot execute policy: Robot is disabled")
        return False

    if policy_state.get_prediction_horizon_length() == 0:
        print("⚠️ No prediction horizon available.")
        return False

    current_joint_angles = data_manager.get_current_joint_angles()
    if current_joint_angles is None:
        return False

    prediction_horizon = policy_state.get_prediction_horizon()
    first_targets = np.degrees([prediction_horizon[jn][0] for jn in JOINT_NAMES])

    # Safety check: Prevent wild jumps
    if np.any(np.abs(current_joint_angles - first_targets) > MAX_SAFETY_THRESHOLD):
        print("⚠️ Cannot execute policy: Robot too far from first predicted action")
        return False

    policy_state.set_ghost_robot_playing(False)
    data_manager.set_teleop_state(False, None, None)
    policy_state.start_policy_execution()

    if policy_state.get_locked_prediction_horizon_length() == 0:
        policy_state.end_policy_execution()
        return False

    data_manager.set_robot_activity_state(RobotActivityState.POLICY_CONTROLLED)
    return True


def end_policy_play(
    data_manager: DataManager,
    policy_state: PolicyState,
    visualizer: Optional[Any],
    status_msg: str,
) -> None:
    """End continuous play and update system state."""
    if policy_state.get_continuous_play_active():
        policy_state.set_continuous_play_active(False)

    if visualizer:
        visualizer.set_ghost_robot_color((1.0, 0.65, 0.0, 0.25))
        visualizer.update_play_policy_button_status(False)
        visualizer.update_policy_status(status_msg)

    policy_state.end_policy_execution()
    data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
    data_manager.set_teleop_state(False, None, None)


def continuous_prediction_worker(
    data_manager, policy, policy_state, visualizer, input_emb, continuous_mode
):
    """Background thread for continuous receding horizon execution."""
    colors = [
        (1.0, 0.65, 0.0, 0.25),
        (0.0, 1.0, 0.0, 0.25),
        (1.0, 0.0, 0.0, 0.25),
        (0.0, 0.0, 1.0, 0.25),
    ]
    c_idx = 0

    print(f"\n🚀 Bootstrapping initial trajectory in '{continuous_mode}' mode...")
    if run_policy(data_manager, policy, policy_state, visualizer, input_emb):
        start_policy_execution(data_manager, policy_state)

    while policy_state.get_continuous_play_active():
        if policy_state.get_locked_prediction_horizon_length() == 0:
            time.sleep(0.01)
            continue

        if continuous_mode == "pipeline":
            # Predict next horizon in the background while moving
            success = run_policy(
                data_manager, policy, policy_state, visualizer, input_emb
            )
            if not success:
                time.sleep(0.05)
                continue

            # Wait until current trajectory is almost finished before hot-swapping
            while policy_state.get_continuous_play_active():
                rem = (
                    policy_state.get_locked_prediction_horizon_length()
                    - policy_state.get_execution_action_index()
                )
                if rem <= 5 or policy_state.get_locked_prediction_horizon_length() == 0:
                    break
                time.sleep(0.01)

        elif continuous_mode == "sequential":
            # Wait for current trajectory to fully finish before querying network
            while policy_state.get_continuous_play_active():
                if (
                    policy_state.get_execution_action_index()
                    >= policy_state.get_locked_prediction_horizon_length()
                ):
                    break
                time.sleep(0.01)

            if not policy_state.get_continuous_play_active():
                break

            success = run_policy(
                data_manager, policy, policy_state, visualizer, input_emb
            )
            if not success:
                time.sleep(0.05)
                continue

        if not policy_state.get_continuous_play_active():
            break

        policy_state.end_policy_execution()
        if start_policy_execution(data_manager, policy_state):
            c_idx = (c_idx + 1) % len(colors)
            if visualizer:
                visualizer.set_ghost_robot_color(colors[c_idx])
        else:
            time.sleep(0.01)


def play_policy(
    data_manager,
    policy,
    policy_state,
    visualizer,
    input_emb,
    continuous_mode="pipeline",
):
    """Toggle for starting/stopping continuous policy mode."""
    if not policy_state.get_continuous_play_active():
        print(f"▶️ Starting {continuous_mode.capitalize()} Mode...")
        policy_state.set_continuous_play_active(True)
        if visualizer:
            visualizer.update_play_policy_button_status(True)
        threading.Thread(
            target=continuous_prediction_worker,
            args=(
                data_manager,
                policy,
                policy_state,
                visualizer,
                input_emb,
                continuous_mode,
            ),
            daemon=True,
        ).start()
    else:
        print("⏹️ Stopping continuous policy execution...")
        end_policy_play(
            data_manager, policy_state, visualizer, "Policy execution stopped"
        )


def policy_execution_thread(
    policy, data_manager, policy_state, robot_controller, visualizer, input_emb
):
    """The thread that continuously reads the locked horizon and sends joint commands."""
    dt_execution = 1.0 / POLICY_EXECUTION_RATE
    last_vis_update = 0.0

    while True:
        start_time = time.time()

        if (
            data_manager.get_robot_activity_state()
            == RobotActivityState.POLICY_CONTROLLED
        ):
            locked_horizon = policy_state.get_locked_prediction_horizon()
            exec_idx = policy_state.get_execution_action_index()
            locked_len = policy_state.get_locked_prediction_horizon_length()

            if exec_idx < locked_len:
                current_angles = data_manager.get_current_joint_angles()

                # Check target pose tracking if in pose-execution mode
                if (
                    exec_idx > 0
                    and current_angles is not None
                    and policy_state.get_execution_mode()
                    == PolicyState.ExecutionMode.TARGETING_POSE
                ):
                    t_start = time.time()
                    while (time.time() - t_start) < TARGETING_POSE_TIME_THRESHOLD:
                        prev_targs = np.degrees(
                            [locked_horizon[jn][exec_idx - 1] for jn in JOINT_NAMES]
                        )
                        if np.any(
                            np.abs(current_angles - prev_targs)
                            <= MAX_ACTION_ERROR_THRESHOLD
                        ):
                            break
                        time.sleep(0.001)

                if all(jn in locked_horizon for jn in JOINT_NAMES):
                    targs_deg = np.degrees(
                        [locked_horizon[jn][exec_idx] for jn in JOINT_NAMES]
                    )
                    data_manager.set_target_joint_angles(targs_deg)
                    if robot_controller.is_robot_enabled():
                        robot_controller.set_target_joint_angles(targs_deg)

                if GRIPPER_NAME in locked_horizon:
                    robot_controller.set_gripper_open_value(
                        locked_horizon[GRIPPER_NAME][exec_idx]
                    )

                policy_state.increment_execution_action_index()
                if visualizer:
                    visualizer.update_policy_status(
                        f"Executing: {exec_idx + 1}/{locked_len}"
                    )
            else:
                if not policy_state.get_continuous_play_active():
                    end_policy_play(
                        data_manager, policy_state, visualizer, "Execution completed"
                    )
                elif (
                    all(jn in locked_horizon for jn in JOINT_NAMES)
                    and robot_controller.is_robot_enabled()
                ):
                    # Hold last position while waiting for background thread to swap horizon
                    robot_controller.set_target_joint_angles(
                        np.degrees([locked_horizon[jn][-1] for jn in JOINT_NAMES])
                    )

        # Throttle visualizer updates to 30Hz
        if visualizer and (time.time() - last_vis_update >= 1.0 / 30.0):
            _update_visualization(data_manager, policy_state, visualizer)
            last_vis_update = time.time()

        time.sleep(max(0, dt_execution - (time.time() - start_time)))


def _update_visualization(
    data_manager: DataManager, policy_state: PolicyState, visualizer: Any
) -> None:
    """Sync the visualizer UI with the DataManager's internal state."""
    current_joints = data_manager.get_current_joint_angles()
    if current_joints is not None:
        visualizer.update_robot_pose(np.radians(current_joints))

    rgb_image = data_manager.get_rgb_image(CAMERA_NAMES[0])
    if rgb_image is not None:
        visualizer.update_rgb_image(rgb_image)

    state = data_manager.get_robot_activity_state()
    if state == RobotActivityState.POLICY_CONTROLLED:
        visualizer.update_ghost_robot_visibility(True)
        t_joints = data_manager.get_target_joint_angles()
        if t_joints is not None:
            visualizer.update_ghost_robot_pose(np.radians(t_joints))
        visualizer.set_run_policy_button_disabled(True)
        visualizer.set_play_policy_button_disabled(False)

    elif state == RobotActivityState.ENABLED and data_manager.get_teleop_active():
        visualizer.update_ghost_robot_visibility(True)
        t_joints = data_manager.get_target_joint_angles()
        if t_joints is not None:
            visualizer.update_ghost_robot_pose(np.radians(t_joints))

    elif (
        policy_state.get_ghost_robot_playing()
        and policy_state.get_prediction_horizon_length() > 0
    ):
        visualizer.set_start_policy_execution_button_disabled(False)
        visualizer.update_ghost_robot_visibility(True)
        g_idx = policy_state.get_ghost_action_index()
        horizon = policy_state.get_prediction_horizon()

        if g_idx < policy_state.get_prediction_horizon_length() and all(
            jn in horizon for jn in JOINT_NAMES
        ):
            visualizer.update_ghost_robot_pose(
                np.array([horizon[jn][g_idx] for jn in JOINT_NAMES])
            )
            policy_state.set_ghost_action_index(
                (g_idx + 1) % policy_state.get_prediction_horizon_length()
            )
        else:
            policy_state.reset_ghost_action_index()
    else:
        visualizer.update_ghost_robot_visibility(False)
        has_horizon = policy_state.get_prediction_horizon_length() > 0
        enabled = state == RobotActivityState.ENABLED

        visualizer.set_start_policy_execution_button_disabled(
            not (enabled and has_horizon)
        )
        visualizer.set_run_policy_button_disabled(not enabled)
        visualizer.set_play_policy_button_disabled(not enabled)
