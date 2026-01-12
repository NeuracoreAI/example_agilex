#!/usr/bin/env python3
"""Simple policy visualization from dataset - single script, no classes."""

import argparse
import random
import sys
import time
from pathlib import Path

import neuracore as nc
import numpy as np
import viser
import yourdfpy
from neuracore_types import (
    BatchedJointData,
    BatchedNCData,
    BatchedParallelGripperOpenAmountData,
    DataType,
    RobotDataSpec,
)
from PIL import Image
from viser.extras import ViserUrdf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import (
    CAMERA_LOGGING_NAME,
    GRIPPER_LOGGING_NAME,
    JOINT_NAMES,
    URDF_PATH,
)

# Parse arguments
parser = argparse.ArgumentParser(
    description="Visualize policy predictions from dataset"
)
parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name")
parser.add_argument(
    "--train-run-name", type=str, default=None, help="Training run name"
)
parser.add_argument("--model-path", type=str, default=None, help="Model file path")
args = parser.parse_args()

if (args.train_run_name is None) == (args.model_path is None):
    parser.error("Exactly one of --train-run-name or --model-path must be provided")

# Connect to NeuraCore
print("🔧 Initializing NeuraCore...")
nc.login()
nc.connect_robot(robot_name="AgileX PiPER", urdf_path=str(URDF_PATH), overwrite=False)

# Load policy
model_input_order = {
    DataType.JOINT_POSITIONS: JOINT_NAMES,
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: [GRIPPER_LOGGING_NAME],
    DataType.RGB_IMAGES: [CAMERA_LOGGING_NAME],
}
model_output_order = {
    DataType.JOINT_TARGET_POSITIONS: JOINT_NAMES,
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: [GRIPPER_LOGGING_NAME],
}

if args.train_run_name:
    print(f"🤖 Loading policy from training run: {args.train_run_name}...")
    policy = nc.policy(
        train_run_name=args.train_run_name,
        device="cuda",
        model_input_order=model_input_order,
        model_output_order=model_output_order,
    )
else:
    print(f"🤖 Loading policy from model file: {args.model_path}...")
    policy = nc.policy(
        model_file=args.model_path,
        device="cuda",
        model_input_order=model_input_order,
        model_output_order=model_output_order,
    )
print("  ✓ Policy loaded")

# Load and synchronize dataset
print(f"🔍 Loading dataset: {args.dataset_name}...")
dataset = nc.get_dataset(args.dataset_name)
print(f"  ✓ Dataset loaded: {len(dataset)} episodes")

robot_data_spec: RobotDataSpec = {
    robot_id: dataset.get_full_data_spec(robot_id) for robot_id in dataset.robot_ids
}

print("🔁 Synchronizing dataset...")
synced_dataset = dataset.synchronize(
    frequency=100,
    robot_data_spec=robot_data_spec,
    prefetch_videos=True,
    max_prefetch_workers=2,
)
print(f"  ✓ Dataset synchronized: {len(synced_dataset)} episodes")

# Setup Viser
print("🖥️  Starting Viser...")
server = viser.ViserServer()
server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

# Load URDF
urdf = yourdfpy.URDF.load(str(URDF_PATH))
urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
urdf_vis.update_cfg(np.zeros(len(JOINT_NAMES)))

# State variables
current_horizon = None
current_action_idx = 0
playing = False


def convert_predictions_to_horizon(
    predictions: dict[DataType, dict[str, BatchedNCData]],
) -> dict[str, list[float]]:
    """Convert predictions to horizon dict."""
    horizon = {}
    if DataType.JOINT_TARGET_POSITIONS in predictions:
        joint_data = predictions[DataType.JOINT_TARGET_POSITIONS]
        for joint_name in JOINT_NAMES:
            if joint_name in joint_data:
                batched = joint_data[joint_name]
                if isinstance(batched, BatchedJointData):
                    values = batched.value[0, :, 0].cpu().numpy().tolist()
                    horizon[joint_name] = values
    if DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in predictions:
        gripper_data = predictions[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS]
        if GRIPPER_LOGGING_NAME in gripper_data:
            batched = gripper_data[GRIPPER_LOGGING_NAME]
            if isinstance(batched, BatchedParallelGripperOpenAmountData):
                values = batched.open_amount[0, :, 0].cpu().numpy().tolist()
                horizon[GRIPPER_LOGGING_NAME] = values
    return horizon


def select_random_state() -> None:
    """Select random state and run policy."""
    global current_horizon, current_action_idx, playing

    # Select random episode and step
    episode_idx = random.randint(0, len(synced_dataset) - 1)
    episode = synced_dataset[episode_idx]
    if len(episode) == 0:
        print(f"⚠️  Episode {episode_idx} is empty")
        return

    step_idx = random.randint(0, len(episode) - 1)
    step = episode[step_idx]
    print(f"📊 Selected episode {episode_idx}, step {step_idx}")

    # Extract joint positions
    joint_positions_dict = {}
    if DataType.JOINT_POSITIONS in step.data:
        joint_data = step.data[DataType.JOINT_POSITIONS]
        for joint_name in JOINT_NAMES:
            if joint_name in joint_data:
                joint_positions_dict[joint_name] = joint_data[joint_name].value

    # Extract gripper
    gripper_value = 1.0
    if DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in step.data:
        gripper_data = step.data[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS]
        if GRIPPER_LOGGING_NAME in gripper_data:
            gripper_value = gripper_data[GRIPPER_LOGGING_NAME].open_amount

    # Extract RGB image
    rgb_image = None
    if DataType.RGB_IMAGES in step.data:
        rgb_data = step.data[DataType.RGB_IMAGES]
        if CAMERA_LOGGING_NAME in rgb_data:
            rgb_image = np.array(rgb_data[CAMERA_LOGGING_NAME].frame)

    if rgb_image is None:
        print("⚠️  No RGB image found")
        return

    # Log to NeuraCore
    nc.log_joint_positions(joint_positions_dict)
    nc.log_parallel_gripper_open_amount(GRIPPER_LOGGING_NAME, gripper_value)
    nc.log_rgb(CAMERA_LOGGING_NAME, rgb_image)

    # Get policy prediction
    print("🎯 Getting policy prediction...")
    predictions = policy.predict(timeout=5)
    current_horizon = convert_predictions_to_horizon(predictions)
    current_action_idx = 0
    playing = True
    print("FINISHED PREDICTION")

    # Save image to file
    image_pil = Image.fromarray(rgb_image)
    image_pil.save("current_image.png")
    print("💾 Saved image to current_image.png")

    # Update robot to initial pose
    joint_positions = np.array([joint_positions_dict[jn] for jn in JOINT_NAMES])
    urdf_vis.update_cfg(joint_positions)

    print(
        f"✅ Prediction received: {len(current_horizon.get(JOINT_NAMES[0], []))} actions"
    )


# Add button
random_button = server.gui.add_button("Random Selection")
random_button.on_click(lambda _: select_random_state())

# Add gripper value display
gripper_handle = server.gui.add_slider(
    "Gripper Open Amount",
    min=0.0,
    max=1.0,
    step=0.01,
    initial_value=0.0,
    disabled=True,  # Read-only
)

# Add frequency control
frequency_handle = server.gui.add_number(
    "Visualization Frequency (Hz)",
    initial_value=100.0,
    min=1.0,
    max=500.0,
    step=1.0,
)

# Select initial state
select_random_state()
# Main loop
try:
    while True:
        start_time = time.time()

        # Update robot visualization
        if (
            playing
            and current_horizon
            and len(current_horizon.get(JOINT_NAMES[0], [])) > 0
        ):
            horizon_length = len(current_horizon[JOINT_NAMES[0]])
            if current_action_idx < horizon_length:
                # Get current action
                joint_config = np.array(
                    [
                        current_horizon[joint_name][current_action_idx]
                        for joint_name in JOINT_NAMES
                    ]
                )
                urdf_vis.update_cfg(joint_config)

                # Update gripper value
                gripper_value = current_horizon[GRIPPER_LOGGING_NAME][
                    current_action_idx
                ]
                gripper_handle.value = round(
                    gripper_value, 2
                )  # Round to 2 decimal places

                # Advance to next action
                current_action_idx = (current_action_idx + 1) % horizon_length

        # Sleep to control update rate
        elapsed = time.time() - start_time
        frequency = max(frequency_handle.value, 0.1)  # Avoid division by zero
        time.sleep(max(0, 1.0 / frequency - elapsed))

except KeyboardInterrupt:
    print("\n👋 Shutting down...")
finally:
    policy.disconnect()
    nc.logout()
