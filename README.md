
# AgileX Piper Robot Teleoperation with Neuracore

<div align="center">
  <img src="docs/demo.gif" width="100%">
</div>

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-join%20chat-7289da.svg)](https://discord.gg/DF5m8V6nbD)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

This repository provides a complete, production-ready framework for integrating the AgileX Piper robotic arm with [Neuracore](https://neuracore.com/). It includes tools for teleoperating the robot using a Meta Quest VR headset, collecting perfectly synchronized demonstration datasets, validating and replaying data, and deploying trained AI policies directly to the hardware.

---

## 📋 Prerequisites

Before starting, ensure your system meets the following requirements:
* **OS/Environment:** Ubuntu/Linux with Conda installed.
* **Python:** Version 3.10 or higher.
* **System Packages:** `sudo apt-get install sox`
* **Hardware:** * AgileX Piper robot arm connected via CAN bus.
    * RealSense camera (or compatible USB webcam).
    * Meta Quest headset (Developer Mode enabled, ADB USB debugging allowed).

---

## ⚙️ Installation

### 1. Clone the Main Repository
First, clone this repository to your workspace:
```bash
git clone git@github.com:NeuracoreAI/example_agilex.git
cd example_agilex

```

### 2. Create the Conda Environment

Create and activate the dedicated Python environment:

```bash
conda env create -f environment.yaml
conda activate piper-teleop

```

### 3. Install the Meta Quest Teleop Package

To keep dependencies organized, clone the Meta Quest teleoperation package **alongside** (in the parent directory of) your `example_agilex` folder, then install it into your active conda environment:

```bash
# Navigate out of example_agilex
cd .. 

# Clone and install the Meta Quest reader
git clone [https://github.com/NeuracoreAI/meta_quest_teleop.git](https://github.com/NeuracoreAI/meta_quest_teleop.git)
cd meta_quest_teleop
pip install -e .

# Return to the main project directory
cd ../example_agilex

```

---

## 🔄 System Architecture & Data Flow

The teleoperation stack operates asynchronously to ensure smooth hardware control and precise data logging:

```text
[ Meta Quest VR Headset ] 
           ↓ (ADB / WiFi)
[ Meta Quest Reader ] (Translates 6D poses & button states)
           ↓ 
[ Pink IK Solver ] (Converts spatial targets to joint angles)
           ↓ 
[ Piper Controller ] (Applies safety limits & smoothing)
           ↓ (CAN Bus)
[ AgileX Piper Robot ] 

```

---

## 🚀 Usage Guide

> ⚠️ **HARDWARE SAFETY NOTE**: This software controls a physical, high-torque industrial robot. **Always keep your hand on the physical emergency stop.** Before running any script, ensure the robot's workspace is entirely clear of obstacles and personnel.

### 1. Tune Teleoperation Parameters

**Script:** `examples/1_tune_teleop_params.py`

Use this script to safely test the robot and dial in your movement preferences before collecting data. It launches a web UI where you can adjust IK weights, motion smoothing (1-Euro filter), and scaling factors on the fly.

```bash
cd examples
python 1_tune_teleop_params.py [--ip-address <quest-ip>] [--ik-config ik_conf/default.yaml]

```

* **Button A:** Enable/Disable robot hardware.
* **Right Grip (Hold):** Dead man's switch. You must hold this to move the robot.
* **Right Trigger:** Open/Close parallel gripper.
* **Button B:** Return robot to home position.
* **Save Config:** Once tuned, use the UI button to save your perfect settings to YAML.

### 2. Collect Teleoperation Data

**Script:** `examples/2_collect_teleop_data_with_neuracore.py`

Once your parameters are tuned, use this script to record demonstrations directly to the Neuracore cloud.

```bash
cd examples
python 2_collect_teleop_data_with_neuracore.py [--dataset-name <name>] [--ik-config ik_conf/default.yaml]

```

* **Right Joystick (Press):** Start/Stop recording an episode.
* *(Note: Passing an existing `--dataset-name` will append new episodes to it. Otherwise, a new timestamped dataset is created).*

### 3. Replay Neuracore Episodes (Hardware Validation)

**Script:** `examples/3_replay_neuracore_episodes.py`

This script downloads a dataset from Neuracore and forces the physical robot to perfectly re-enact the recorded trajectory.

> 🛑 **CRITICAL SAFETY WARNINGS**:
> * **Movement:** The robot will move identically to the recorded data. **Be ready to press the emergency stop.** Pressing `Ctrl+C` will gracefully halt the script and cut power to the motors after 5 seconds.
> * **Episode Indexing:** The index starts at 0. If you want to replay the episode labeled "3" in the Neuracore web dashboard, you must pass `--episode-index 2`.
> * **Session Resets:** You may need to fully restart the script/robot if you wish to run multiple different replays back-to-back.
> 
> 

```bash
cd examples
python 3_replay_neuracore_episodes.py --dataset-name <name> [--episode-index <index>] [--frequency 20]

```

* **`--frequency`:** Playback speed in Hz. The default is `20`. Passing `0` plays the data aperiodically (exactly as the raw packets were recorded).
* **`--episode-index`:** Set to `-1` to replay all episodes in the dataset sequentially.

### 4. Rollout Neuracore Policy (Full GUI)

**Script:** `examples/4_rollout_neuracore_policy.py`

Deploy a trained AI model to the robot. This script opens a Viser 3D dashboard where you can manually pilot the robot to a starting state using the VR controller, preview the AI's predicted trajectory as a "ghost robot," and then execute the AI's actions.

```bash
# Load from a local model file
cd examples
python 4_rollout_neuracore_policy.py --model-path <path-to-model.nc.zip>

# Load from a cloud training run
python 4_rollout_neuracore_policy.py --train-run-name <run-name>

# Connect to an active remote inference server
python 4_rollout_neuracore_policy.py --remote-endpoint-name <endpoint-name>

```

### 5. Rollout Neuracore Policy (Minimal / Headless)

**Script:** `examples/5_rollout_neuracore_policy_minimal.py`

A lightweight, terminal-only version of the rollout script. It strips away the 3D GUI and Meta Quest tracking, making it perfect for rapid, automated deployments in constrained compute environments.

```bash
cd examples
python 5_rollout_neuracore_policy_minimal.py --train-run-name <run-name>

```

### 6. Visualize Policy Offline

**Script:** `examples/6_visualize_policy_from_dataset.py`

The safest way to validate a model. This script pulls camera and joint data from a recorded dataset, feeds it to your AI policy, and renders the AI's predictions in a 3D web simulation. No physical robot is required.

```bash
cd examples
python 6_visualize_policy_from_dataset.py --dataset-name <dataset-name> --train-run-name <run-name>

```

---

## 🛠️ Configuration (`ik_conf/default.yaml`)

To ensure maximum flexibility, all hardware tuning parameters have been extracted into YAML configuration files located in the `ik_conf/` directory.

You can modify `ik_conf/default.yaml` (or create your own profiles like `heavy_payload.yaml`) to adjust:

* **IK Parameters:** Position vs. orientation costs, joint damping, and posture preferences.
* **Filter Parameters:** 1-Euro smoothing coefficients to eliminate VR controller jitter.
* **Teleop Parameters:** Spatial translation/rotation scaling (e.g., mapping 1cm of hand movement to 2cm of robot movement).

Simply pass `--ik-config ik_conf/your_config.yaml` to any script to load a specific profile.

---

## 📁 Project Structure

```text
example_agilex/
├── docs/                   # Documentation assets
│   └── demo.gif            # Main teleoperation demonstration file
├── examples/               # Task-specific execution scripts
│   ├── 1_tune_teleop_params.py
│   ├── 2_collect_teleop_data_with_neuracore.py
│   ├── 3_replay_neuracore_episodes.py
│   ├── 4_rollout_neuracore_policy.py
│   ├── 5_rollout_neuracore_policy_minimal.py
│   ├── 6_visualize_policy_from_dataset.py
│   ├── 7_teleop_with_pedal.py
│   ├── combine_code.py     # Source aggregation helper script
│   ├── ik_conf/            # Runtime parameter configurations
│   │   └── default.yaml    # Tunable profile (IK, smoothing, and scaling targets)
│   └── common/             # Monolithic background framework layers
│       ├── config_parser.py
│       ├── configs.py      # Hardcoded system constants
│       ├── data_manager.py # Thread-safe global state machine
│       ├── dataset_helpers.py
│       ├── foot_pedal.py   # Keystroke mapping for foot pedals
│       ├── one_euro_filter.py
│       ├── policy_actions.py
│       ├── policy_helpers.py
│       ├── policy_state.py
│       ├── robot_visualizer.py
│       ├── shared_actions.py
│       ├── states.py
│       ├── system_bootstrap.py
│       ├── utils.py
│       ├── visualizer_core.py
│       ├── visualizer_gui.py
│       └── threads/        # Parallel processing workers
│           ├── camera_usb.py
│           ├── ik_solver.py
│           ├── joint_state.py
│           ├── quest_reader.py
│           └── realsense_camera.py
├── piper_description/      # Physical specifications of the manipulator
│   ├── meshes/             # Component link geometry files (base_link.STL to link8.STL)
│   └── urdf/               # AgileX Piper robot model definition file (.urdf)
├── scripts/                # Isolated automation and helper scripts
│   ├── oculus/             # Standalone VR tracking debug tools
│   │   └── monitor_hand_movement.py
│   └── piper/              # Hardware utilities
│       ├── can_activate.sh # Bash routine to initialize the 'can0' interface
│       └── piper_gui_control.py
├── environment.yaml        # Conda dependency definitions
├── pink_ik_solver.py       # Intermediary task space inverse kinematics layer
├── piper_controller.py     # Native CAN-bus mapping driver for the robot
└── vectorised_posture_task.py

```

---

## 🔧 Troubleshooting

* **ADB / Meta Quest Permission Errors:** If the scripts immediately crash complaining about Meta Quest access, put the headset on, look for a "USB Detected" popup, and click "Allow".
* **CAN Bus / Robot Communication Issues:** If the script hangs on `Initializing robot on can0...`, verify your CAN interface is active. Run `ip link show can0`. If it is down, activate it using:
```bash
bash scripts/piper/can_activate.sh can0 1000000

```


* **Remote Endpoint Crashes:**
If Scripts 4, 5, or 6 throw an `EndpointError`, verify that your deployment server is actually set to "Active" in the Neuracore web dashboard.

---

## 📄 License & Support

See the [LICENSE](https://www.google.com/search?q=LICENSE) file for open-source terms. For questions, bug reports, or feature requests, please open an Issue on GitHub or join our community Discord server.
