#!/usr/bin/env python3
"""Simple policy visualization from dataset.

Loads a policy and a dataset, and randomly selects a state
from the dataset to run the policy with and visualize the results.
"""

import argparse
import random
import sys
import time
import traceback
from pathlib import Path

import neuracore as nc
import numpy as np
import viser
import yourdfpy
from neuracore.core.utils.robot_data_spec_utils import (
    merge_cross_embodiment_description,
)
from neuracore_types import DataType
from PIL import Image
from viser.extras import ViserUrdf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import JOINT_NAMES, POLICY_EXECUTION_RATE, URDF_PATH
from common.policy_helpers import (
    DEFAULT_ROBOT_NAME,
    convert_predictions_to_horizon,
    embodiment_names_ordered,
    get_policy_embodiments,
    gripper_open_at_index,
    horizon_length,
    log_sync_step_for_policy,
    print_policy_embodiments,
    urdf_cfg_from_horizon,
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
    "--robot-name",
    type=str,
    default=DEFAULT_ROBOT_NAME,
    help="Neuracore robot name (policy embodiment resolution).",
)
parser.add_argument(
    "--frequency",
    type=int,
    default=POLICY_EXECUTION_RATE,
    help="Frequency of visualization",
)
args = parser.parse_args()

# Connect to Neuracore
print("🔧 Initializing Neuracore...")
nc.login()
nc.connect_robot(robot_name=args.robot_name, urdf_path=str(URDF_PATH), overwrite=False)

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
        robot_name=args.robot_name,
    )
else:
    print(f"🤖 Loading policy from model file: {args.model_path}...")
    policy = nc.policy(
        model_file=args.model_path,
        device="cuda",
        robot_name=args.robot_name,
    )
print("  ✓ Policy loaded")
input_embodiment_description, output_embodiment_description = get_policy_embodiments(
    policy
)
print_policy_embodiments(input_embodiment_description, output_embodiment_description)

output_gripper_names = None
if output_embodiment_description is not None:
    gripper_spec = output_embodiment_description.get(
        DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS
    )
    if gripper_spec is not None:
        output_gripper_names = embodiment_names_ordered(gripper_spec)

# Load and synchronize dataset
print(f"🔍 Loading dataset: {args.dataset_name}...")
dataset = nc.get_dataset(args.dataset_name)
print(f"  ✓ Dataset loaded: {len(dataset)} episodes")

print("🔁 Building cross_embodiment_union for synchronization...")
input_cross_embodiment_description = {
    robot_id: input_embodiment_description for robot_id in dataset.robot_ids
}
output_cross_embodiment_description = (
    {robot_id: output_embodiment_description for robot_id in dataset.robot_ids}
    if output_embodiment_description is not None
    else {}
)
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


def select_random_state() -> None:
    """Select random state and run policy."""
    global current_horizon, current_action_idx, playing, rgb_gui_handle

    episode_idx = random.randint(0, len(synced_dataset) - 1)
    episode = synced_dataset[episode_idx]
    if len(episode) == 0:
        print(f"⚠️  Episode {episode_idx} is empty")
        return

    step_idx = random.randint(0, len(episode) - 1)
    step = episode[step_idx]
    print(f"📊 Selected episode {episode_idx}, step {step_idx}")

    if not log_sync_step_for_policy(step, input_embodiment_description):
        print("⚠️  Step has no data for policy input channels")
        return

    if hasattr(step, "data"):
        rgb_data = step.data.get(DataType.RGB_IMAGES, {})
        for _camera_name, frame in rgb_data.items():
            rgb_image = np.array(frame.frame)
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
            break

    print("🎯 Getting policy prediction...")
    start_time = time.time()
    try:
        predictions = policy.predict(timeout=60)
    except nc.EndpointError as e:
        print(f"✗ Failed to get policy prediction: {e}")
        traceback.print_exc()
        return
    duration = time.time() - start_time
    current_horizon = convert_predictions_to_horizon(predictions)
    current_action_idx = 0
    playing = True
    print(f"FINISHED PREDICTION in {duration:.3f} s")

    joint_cfg = urdf_cfg_from_horizon(current_horizon or {}, 0)
    if joint_cfg is not None:
        urdf_vis.update_cfg(joint_cfg)

    print(f"✅ Prediction received: {horizon_length(current_horizon or {})} actions")


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
    disabled=True,
)

# Add frequency control
frequency_handle = server.gui.add_number(
    "Visualization Frequency (Hz)",
    initial_value=args.frequency,
    min=1.0,
    max=500.0,
    step=1.0,
)

select_random_state()
try:
    while True:
        start_time = time.time()

        h_len = horizon_length(current_horizon or {})
        if playing and current_horizon and h_len > 0:
            if current_action_idx < h_len:
                joint_cfg = urdf_cfg_from_horizon(current_horizon, current_action_idx)
                if joint_cfg is not None:
                    urdf_vis.update_cfg(joint_cfg)
                    nc.log_joint_positions(
                        {jn: float(joint_cfg[i]) for i, jn in enumerate(JOINT_NAMES)}
                    )

                gripper_value = gripper_open_at_index(
                    current_horizon,
                    current_action_idx,
                    gripper_names=output_gripper_names,
                )
                if gripper_value is not None:
                    gripper_handle.value = round(gripper_value, 2)

                current_action_idx = (current_action_idx + 1) % h_len

        elapsed = time.time() - start_time
        frequency = max(frequency_handle.value, 0.1)
        time.sleep(max(0, 1.0 / frequency - elapsed))

except KeyboardInterrupt:
    print("\n👋 Shutting down...")
finally:
    policy.disconnect()
    nc.logout()
