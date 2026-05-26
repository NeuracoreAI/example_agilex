#!/usr/bin/env python3
"""Simple policy visualization from dataset."""

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
from PIL import Image
from viser.extras import ViserUrdf

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import JOINT_NAMES, POLICY_EXECUTION_RATE, URDF_PATH
from common.dataset_helpers import load_and_sync_dataset
from common.policy_helpers import (
    DEFAULT_ROBOT_NAME, convert_predictions_to_horizon, embodiment_names_ordered,
    get_policy_embodiments, gripper_open_at_index, horizon_length,
    log_sync_step_for_policy, print_policy_embodiments, urdf_cfg_from_horizon,
)
from neuracore_types import DataType

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize policy predictions")
    parser.add_argument("--dataset-name", type=str, required=True)
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument("--train-run-name", type=str, default=None)
    policy_group.add_argument("--model-path", type=str, default=None)
    policy_group.add_argument("--remote-endpoint-name", type=str, default=None)
    parser.add_argument("--robot-name", type=str, default=DEFAULT_ROBOT_NAME)
    parser.add_argument("--frequency", type=int, default=POLICY_EXECUTION_RATE)
    args = parser.parse_args()

    nc.login()
    nc.connect_robot(robot_name=args.robot_name, urdf_path=str(URDF_PATH), overwrite=False)

    if args.remote_endpoint_name:
        policy = nc.policy_remote_server(args.remote_endpoint_name)
    elif args.train_run_name:
        policy = nc.policy(train_run_name=args.train_run_name, device="cuda", robot_name=args.robot_name)
    else:
        policy = nc.policy(model_file=args.model_path, device="cuda", robot_name=args.robot_name)

    input_emb, output_emb = get_policy_embodiments(policy)
    print_policy_embodiments(input_emb, output_emb)

    out_grippers = embodiment_names_ordered(output_emb[DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS]) if output_emb and DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS in output_emb else None

    input_mods = list(input_emb.keys())
    output_mods = list(output_emb.keys()) if output_emb else []
    synced_dataset = load_and_sync_dataset(args.dataset_name, args.frequency, input_mods, output_mods, prefetch_videos=True)

    print("🖥️  Starting Viser...")
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, yourdfpy.URDF.load(str(URDF_PATH)), root_node_name="/robot")
    urdf_vis.update_cfg(np.zeros(len(JOINT_NAMES)))

    state = {"horizon": None, "action_idx": 0, "playing": False, "rgb_handle": None}

    def select_random_state() -> None:
        episode = synced_dataset[random.randint(0, len(synced_dataset) - 1)]
        if not len(episode): return
        step = episode[random.randint(0, len(episode) - 1)]

        if not log_sync_step_for_policy(step, input_emb): return

        rgb_data = getattr(step, "data", {}).get(DataType.RGB_IMAGES, {})
        for _, frame in rgb_data.items():
            rgb_arr = np.array(frame.frame)
            Image.fromarray(rgb_arr).save("current_image.png")
            if state["rgb_handle"] is None:
                state["rgb_handle"] = server.gui.add_image(rgb_arr, label="RGB (current step)", format="jpeg")
            else:
                state["rgb_handle"].image = rgb_arr
            break

        try:
            state["horizon"] = convert_predictions_to_horizon(policy.predict(timeout=60))
            state["action_idx"] = 0
            state["playing"] = True
            
            joint_cfg = urdf_cfg_from_horizon(state["horizon"], 0)
            if joint_cfg is not None: urdf_vis.update_cfg(joint_cfg)
        except Exception as e:
            print(f"✗ Failed prediction: {e}")

    server.gui.add_button("Random Selection").on_click(lambda _: select_random_state())
    gripper_handle = server.gui.add_slider("Gripper Open Amount", min=0.0, max=1.0, step=0.01, initial_value=0.0, disabled=True)
    freq_handle = server.gui.add_number("Visualization Frequency (Hz)", initial_value=args.frequency, min=1.0, max=500.0, step=1.0)

    select_random_state()

    try:
        while True:
            start_time = time.time()
            h_len = horizon_length(state["horizon"] or {})

            if state["playing"] and state["horizon"] and h_len > 0:
                if state["action_idx"] < h_len:
                    j_cfg = urdf_cfg_from_horizon(state["horizon"], state["action_idx"])
                    if j_cfg is not None:
                        urdf_vis.update_cfg(j_cfg)
                        nc.log_joint_positions({jn: float(j_cfg[i]) for i, jn in enumerate(JOINT_NAMES)})

                    g_val = gripper_open_at_index(state["horizon"], state["action_idx"], out_grippers)
                    if g_val is not None: gripper_handle.value = round(g_val, 2)
                    state["action_idx"] = (state["action_idx"] + 1) % h_len

            time.sleep(max(0, 1.0 / max(freq_handle.value, 0.1) - (time.time() - start_time)))

    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        policy.disconnect()
        nc.logout()