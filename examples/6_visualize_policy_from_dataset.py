#!/usr/bin/env python3
"""
Offline Policy Visualization and Validation.

This script acts as a safe, simulated testing ground for your trained AI policies.
Instead of running on the physical robot, it:
    1. Loads a synchronized teleoperation dataset from Neuracore.
    2. Selects a random step (observation) from the dataset.
    3. Feeds that real-world camera and joint data into the trained policy.
    4. Renders the AI's predicted future action horizon in a 3D Viser web UI.

It is highly recommended to run this script to visually validate a model's sanity
before deploying it to the physical AgileX Piper arm (via Scripts 4 or 5).

Usage Examples:
    python 6_visualize_policy_from_dataset.py --dataset-name my_dataset --model-path ./model.nc.zip
    python 6_visualize_policy_from_dataset.py --dataset-name my_dataset --train-run-name cloud_run
"""

import argparse
import logging
import random
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Suppress Noisy WebRTC/STUN Networking Errors from Viser
# ---------------------------------------------------------------------------
# Viser attempts to use STUN servers for WebRTC streaming. If a firewall/VPN
# blocks it, it throws massive 401 errors before falling back to WebSockets.
# This silences those unhandled asyncio exceptions so the terminal stays clean.
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("aioice").setLevel(logging.CRITICAL)
logging.getLogger("aiortc").setLevel(logging.CRITICAL)

import neuracore as nc
import numpy as np
import viser
import yourdfpy
from PIL import Image
from viser.extras import ViserUrdf

# ---------------------------------------------------------------------------
# Path Configuration & Local Imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import (
    CAMERA_NAMES,
    GRIPPER_NAME,
    JOINT_NAMES,
    POLICY_EXECUTION_RATE,
    URDF_PATH,
)
from common.dataset_helpers import load_and_sync_dataset
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
from neuracore_types import DataType

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. Argument Parsing
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Visualize policy predictions offline using dataset states."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Neuracore dataset to draw observations from.",
    )

    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument(
        "--train-run-name", type=str, default=None, help="Cloud training run name."
    )
    policy_group.add_argument(
        "--model-path", type=str, default=None, help="Path to local .nc.zip model file."
    )
    policy_group.add_argument(
        "--remote-endpoint-name",
        type=str,
        default=None,
        help="Active remote inference endpoint.",
    )

    parser.add_argument("--robot-name", type=str, default=DEFAULT_ROBOT_NAME)
    parser.add_argument("--frequency", type=int, default=POLICY_EXECUTION_RATE)
    args = parser.parse_args()

    print("=" * 60 + "\nOFFLINE POLICY VISUALIZATION\n" + "=" * 60)

    # ---------------------------------------------------------
    # 2. Neuracore Initialization & Policy Loading
    # ---------------------------------------------------------
    nc.login()
    nc.connect_robot(
        robot_name=args.robot_name, urdf_path=str(URDF_PATH), overwrite=False
    )

    if args.remote_endpoint_name:
        print(
            f"\n🤖 Connecting to remote policy endpoint: {args.remote_endpoint_name}..."
        )
        try:
            policy = nc.policy_remote_server(args.remote_endpoint_name)
        except Exception:
            print("\n" + "!" * 60)
            print(f"❌ ENDPOINT NOT ACTIVE: '{args.remote_endpoint_name}'")
            print("!" * 60)
            print(
                "The script successfully reached Neuracore, but the remote server is down."
            )
            print("\nPLEASE FOLLOW THESE STEPS:")
            print("  1. Open your browser and go to the Neuracore website/dashboard.")
            print(
                f"  2. Locate your deployment endpoint named '{args.remote_endpoint_name}'."
            )
            print("  3. Click 'Deploy' or 'Activate' to spin up the cloud server.")
            print(
                "  4. Wait for the status to show as 'Active', then rerun this script."
            )
            print("!" * 60 + "\n")
            sys.exit(1)

    elif args.train_run_name:
        print(f"\n🤖 Loading policy from cloud training run: {args.train_run_name}...")
        policy = nc.policy(
            train_run_name=args.train_run_name,
            device="cuda",
            robot_name=args.robot_name,
        )
    else:
        print(f"\n🤖 Loading policy from local model: {args.model_path}...")
        policy = nc.policy(
            model_file=args.model_path, device="cuda", robot_name=args.robot_name
        )

    # ---------------------------------------------------------
    # 3. Embodiment Extraction & Fallback
    # ---------------------------------------------------------
    try:
        input_emb, output_emb = get_policy_embodiments(policy)
    except AttributeError:
        print(
            "\n⚠️  Could not dynamically extract embodiments. Using default Piper configuration..."
        )
        input_emb = {
            DataType.JOINT_POSITIONS: {i: f"joint{i+1}" for i in range(6)},
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: {0: GRIPPER_NAME},
            DataType.RGB_IMAGES: {0: CAMERA_NAMES[0]},
        }
        output_emb = {
            DataType.JOINT_TARGET_POSITIONS: {i: f"joint{i+1}" for i in range(6)},
            DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS: {0: GRIPPER_NAME},
        }

    print_policy_embodiments(input_emb, output_emb)

    out_grippers = (
        embodiment_names_ordered(
            output_emb[DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS]
        )
        if output_emb and DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS in output_emb
        else None
    )

    # ---------------------------------------------------------
    # 4. Dataset Synchronization
    # ---------------------------------------------------------
    input_mods = list(input_emb.keys())
    output_mods = list(output_emb.keys()) if output_emb else []

    synced_dataset = load_and_sync_dataset(
        args.dataset_name, args.frequency, input_mods, output_mods, prefetch_videos=True
    )

    if len(synced_dataset) == 0:
        print("❌ Error: The synchronized dataset is empty. Exiting.")
        sys.exit(1)

    # ---------------------------------------------------------
    # 5. Viser 3D UI Setup
    # ---------------------------------------------------------
    print("\n🖥️  Starting Viser simulation environment...")
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

    urdf_vis = ViserUrdf(
        server, yourdfpy.URDF.load(str(URDF_PATH)), root_node_name="/robot"
    )
    urdf_vis.update_cfg(np.zeros(len(JOINT_NAMES)))

    state = {"horizon": None, "action_idx": 0, "playing": False, "rgb_handle": None}

    def select_random_state() -> None:
        """Pulls a random observation from the dataset and queries the AI policy."""
        episode = synced_dataset[random.randint(0, len(synced_dataset) - 1)]
        if not len(episode):
            return
        step = episode[random.randint(0, len(episode) - 1)]

        if not log_sync_step_for_policy(step, input_emb):
            return

        rgb_data = getattr(step, "data", {}).get(DataType.RGB_IMAGES, {})
        for _, frame in rgb_data.items():
            rgb_arr = np.array(frame.frame)
            Image.fromarray(rgb_arr).save("current_image.png")

            if state["rgb_handle"] is None:
                state["rgb_handle"] = server.gui.add_image(
                    rgb_arr, label="RGB Observation", format="jpeg"
                )
            else:
                state["rgb_handle"].image = rgb_arr
            break

        try:
            print("🧠 Querying AI model for prediction horizon...")
            predictions = policy.predict(timeout=60)
            state["horizon"] = convert_predictions_to_horizon(predictions)
            state["action_idx"] = 0
            state["playing"] = True

            joint_cfg = urdf_cfg_from_horizon(state["horizon"], 0)
            if joint_cfg is not None:
                urdf_vis.update_cfg(joint_cfg)

            print(
                f"✓ Prediction successful. Horizon length: {horizon_length(state['horizon'])}"
            )
        except Exception as e:
            print(f"❌ Failed prediction inference: {e}")

    server.gui.add_button("🎲 Pull Random Observation").on_click(
        lambda _: select_random_state()
    )
    gripper_handle = server.gui.add_slider(
        "Predicted Gripper State",
        min=0.0,
        max=1.0,
        step=0.01,
        initial_value=0.0,
        disabled=True,
    )
    freq_handle = server.gui.add_number(
        "Simulation Speed (Hz)",
        initial_value=args.frequency,
        min=1.0,
        max=500.0,
        step=1.0,
    )

    select_random_state()
    print(
        "\n🚀 System Online! Open http://localhost:8080 to view the simulation. Press Ctrl+C to exit.\n"
    )

    # ---------------------------------------------------------
    # 6. Main Visualization Loop
    # ---------------------------------------------------------
    try:
        while True:
            start_time = time.time()
            h_len = horizon_length(state["horizon"] or {})

            if state["playing"] and state["horizon"] and h_len > 0:
                if state["action_idx"] < h_len:
                    j_cfg = urdf_cfg_from_horizon(state["horizon"], state["action_idx"])
                    if j_cfg is not None:
                        urdf_vis.update_cfg(j_cfg)
                        nc.log_joint_positions(
                            {jn: float(j_cfg[i]) for i, jn in enumerate(JOINT_NAMES)}
                        )

                    g_val = gripper_open_at_index(
                        state["horizon"], state["action_idx"], out_grippers
                    )
                    if g_val is not None:
                        gripper_handle.value = round(g_val, 2)

                    state["action_idx"] = (state["action_idx"] + 1) % h_len

            time.sleep(
                max(0.0, 1.0 / max(freq_handle.value, 0.1) - (time.time() - start_time))
            )

    except KeyboardInterrupt:
        print("\n👋 Interrupt received. Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Unhandled error in rendering loop: {e}")
        traceback.print_exc()

    # ---------------------------------------------------------
    # 7. Cleanup
    # ---------------------------------------------------------
    finally:
        print("\n🧹 Severing backend connections...")
        policy.disconnect()
        nc.logout()
        print("👋 Offline validation complete.")
