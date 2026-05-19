#!/usr/bin/env python3
"""Simple policy visualization from dataset.

Loads a policy and a dataset, and randomly selects a state
from the dataset to run the policy with and visualize the results.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import neuracore as nc
import numpy as np
import viser
import yourdfpy
from neuracore.core.utils.robot_data_spec_utils import (
    merge_cross_embodiment_description,
)
from neuracore.ml.preprocessing.methods.resize_pad import ResizePad
from neuracore.ml.utils.preprocessing_utils import PreprocessingConfiguration
from neuracore_types import (
    BatchedJointData,
    BatchedNCData,
    BatchedParallelGripperOpenAmountData,
    DataType,
    EmbodimentDescription,
)
from PIL import Image
from viser.extras import ViserUrdf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import (
    CAMERA_NAMES,
    GRIPPER_NAME,
    JOINT_NAMES,
    POLICY_EXECUTION_RATE,
    URDF_PATH,
)

# Parse arguments
parser = argparse.ArgumentParser(
    description="Visualize policy predictions from dataset"
)
parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name")
policy_group = parser.add_mutually_exclusive_group(required=True)
policy_group.add_argument(
    "--train-run-name", type=str, default=None, help="Training run name"
)
policy_group.add_argument(
    "--model-path", type=str, default=None, help="Model file path"
)
policy_group.add_argument(
    "--remote-endpoint-name",
    type=str,
    default=None,
    help="Name of remote Neuracore policy endpoint to use instead of a local policy.",
)
parser.add_argument(
    "--frequency",
    type=int,
    default=POLICY_EXECUTION_RATE,
    help="Frequency of visualization",
)
parser.add_argument(
    "--robot-name",
    type=str,
    default="AgileX PiPER",
    help="Name of the robot to use",
)
args = parser.parse_args()

# Connect to Neuracore
print("🔧 Initializing Neuracore...")
nc.login()
nc.connect_robot(robot_name=args.robot_name, urdf_path=str(URDF_PATH), overwrite=False)

input_embodiment_description: EmbodimentDescription = {
    DataType.JOINT_POSITIONS: JOINT_NAMES,
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: [GRIPPER_NAME],
    DataType.RGB_IMAGES: [CAMERA_NAMES[0]],
}
output_embodiment_description: EmbodimentDescription = {
    DataType.JOINT_TARGET_POSITIONS: JOINT_NAMES,
    DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS: [GRIPPER_NAME],
}


input_preprocessing_config: PreprocessingConfiguration = {
    DataType.RGB_IMAGES: [ResizePad(size=(224, 224))],
}

if args.remote_endpoint_name:
    print(f"🤖 Connecting to remote policy endpoint: {args.remote_endpoint_name}...")
    try:
        policy = nc.policy_remote_server(args.remote_endpoint_name)
    except nc.EndpointError:
        print(
            f"❌ Endpoint '{args.remote_endpoint_name}' not available. "
            "Please start it from the Neuracore dashboard."
        )
        sys.exit(1)
elif args.train_run_name:
    print(f"🤖 Loading policy from training run: {args.train_run_name}...")
    policy = nc.policy(
        train_run_name=args.train_run_name,
        device="cuda",
        input_embodiment_description=input_embodiment_description,
        output_embodiment_description=output_embodiment_description,
        input_preprocessing_config=input_preprocessing_config,
        robot_name=args.robot_name,
    )
else:
    print(f"🤖 Loading policy from model file: {args.model_path}...")
    policy = nc.policy(
        model_file=args.model_path,
        device="cuda",
        input_embodiment_description=input_embodiment_description,
        output_embodiment_description=output_embodiment_description,
        input_preprocessing_config=input_preprocessing_config,
        robot_name=args.robot_name,
    )
print("  ✓ Policy loaded")

# Load and synchronize dataset
print(f"🔍 Loading dataset: {args.dataset_name}...")
dataset = nc.get_dataset(args.dataset_name)
print(f"  ✓ Dataset loaded: {len(dataset)} episodes")

input_cross_embodiment_description = {args.robot_name: input_embodiment_description}
output_cross_embodiment_description = {args.robot_name: output_embodiment_description}
cross_embodiment_union = merge_cross_embodiment_description(
    input_cross_embodiment_description,
    output_cross_embodiment_description,
)

print("🔁 Synchronizing dataset...")
synced_dataset = dataset.synchronize(
    frequency=args.frequency,
    cross_embodiment_union=cross_embodiment_union,
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
rgb_gui_handle = None


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
    if DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS in predictions:
        gripper_data = predictions[DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS]
        if GRIPPER_NAME in gripper_data:
            batched = gripper_data[GRIPPER_NAME]
            if isinstance(batched, BatchedParallelGripperOpenAmountData):
                values = batched.open_amount[0, :, 0].cpu().numpy().tolist()
                horizon[GRIPPER_NAME] = values
    return horizon


def select_random_state() -> None:
    """Select random state and run policy."""
    global current_horizon, current_action_idx, playing, rgb_gui_handle

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
        # Log to Neuracore for visualization
        nc.log_joint_positions(joint_positions_dict)

    # Extract gripper
    gripper_value = 1.0
    if DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in step.data:
        gripper_data = step.data[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS]
        if GRIPPER_NAME in gripper_data:
            gripper_value = gripper_data[GRIPPER_NAME].open_amount
            # Log to Neuracore for visualization
            nc.log_parallel_gripper_open_amount(GRIPPER_NAME, gripper_value)

    # Extract RGB image
    rgb_image = None
    if DataType.RGB_IMAGES in step.data:
        rgb_data = step.data[DataType.RGB_IMAGES]
        if CAMERA_NAMES[0] in rgb_data:
            rgb_image = np.array(rgb_data[CAMERA_NAMES[0]].frame)
            image_pil = Image.fromarray(rgb_image)
            image_pil.save("current_image.png")
            print("💾 Saved image to current_image.png")
            if rgb_gui_handle is None:
                rgb_gui_handle = server.gui.add_image(
                    rgb_image,
                    label="RGB (current step)",
                    format="jpeg",
                    jpeg_quality=85,
                )
            else:
                rgb_gui_handle.image = rgb_image
            # Log to Neuracore for visualization
            nc.log_rgb(CAMERA_NAMES[0], rgb_image)
    # Get policy prediction
    print("🎯 Getting policy prediction...")
    start_time = time.time()
    try:
        predictions = policy.predict(timeout=60)
    except nc.EndpointError as e:
        print(f"✗ Failed to get policy prediction: {e}")
        import traceback

        traceback.print_exc()
        return
    duration = time.time() - start_time
    current_horizon = convert_predictions_to_horizon(predictions)
    current_action_idx = 0
    playing = True
    print(f"FINISHED PREDICTION in {duration:.3f} s")

    # Update robot to initial pose from first step in the horizon

    joint_positions = np.array([current_horizon[jn][0] for jn in JOINT_NAMES])
    urdf_vis.update_cfg(joint_positions)

    horizon_len = len(current_horizon.get(JOINT_NAMES[0], []))
    print(f"✅ Prediction received: {horizon_len} actions")


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
    initial_value=args.frequency,
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

                # Log to Neuracore for visualization
                # NOTE: we log to joint positions instead of joint target positions
                # because the latter is not visualized by Neuracore
                joint_config_dict = {
                    jn: joint_config[i] for i, jn in enumerate(JOINT_NAMES)
                }
                nc.log_joint_positions(joint_config_dict)

                # Update gripper value
                gripper_value = current_horizon[GRIPPER_NAME][current_action_idx]
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
