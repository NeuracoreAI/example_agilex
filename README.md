# AgileX Piper Robot Teleoperation with NeuraCore

This project is a complete example showcasing how to use Neuracore with the AgileX Piper robot. The project provides examples teleoperating the AgileX Piper robot using a Meta Quest controller, collecting demonstration data with [Neuracore](https://neuracore.com/), deploying trained policies, and an easy interface to tune most of the associated parameters.

## Prerequisites

- Python 3.10
- Conda (for environment management)
- Meta Quest device setup (see `meta_quest_teleop/README.md` for details)
- Realsense camera 

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:NeuracoreAI/example_agilex.git
cd example_agilex
```

### 2. Create Conda Environment

Create and activate the conda environment:

```bash
conda env create -f environment.yaml
conda activate piper-teleop
```

### 3. Install Meta Quest Teleop Package

Install the Meta Quest teleoperation package:

```bash
cd meta_quest_teleop
pip install -e .
cd ..
```

## Data Flow

The teleoperation system follows this data flow:

```
Meta Quest Controller
    ↓
Meta Quest Reader (originally by RAIL, improved by Neuracore)
    ↓
Piper Controller (improved by Neuracore)
    ↓
Pink IK Solver
    ↓
AgileX Piper Robot
```

**Components**:
- **Meta Quest Reader**: Reads controller pose and button inputs from the Meta Quest device
- **Piper Controller**: Manages robot state and sends joint commands via CAN interface
- **Pink IK Solver**: Solves inverse kinematics to convert end-effector poses from the meta quest into joint angles

## Usage

### 1. Tune Teleoperation Parameters

**Script**: `examples/1_tune_teleop_params.py`

This script allows you to teleoperate the robot and tune control parameters using a GUI before you start collecting data. Useful for finding the best hyperparameters (optimal IK parameters, controller filter settings, and scaling factors) for your style of teleoperation. 

**NOTE:** The provided default hyperparameters were tuned on the AgileX Piper robot with a meta quest 3 and worked well for our demos. Make sure to
copy your values into the `configs.py` if you change them in the GUI and find more suitable parameters.

```bash
python examples/1_tune_teleop_params.py [--ip-address <quest-ip>]
```

**Arguments**:
- `--ip-address`: IP address of Meta Quest device (optional). Only needed when using WiFi connection. If not provided, defaults to auto-discovery via USB.

**Controls**:
- **Button A**: Enable/disable robot
- **Right Grip**: Activate teleoperation (dead man's switch)
- **Right Trigger**: Close/open gripper
- **Button B**: Move robot to home position
- **GUI**: Adjust IK parameters, filter settings, scaling factors

### 2. Collect Teleoperation Data

**Script**: `examples/2_collect_teleop_data_with_neuracore.py`

Now that you have a well-tuned setup, you can use this script to record teleoperation demonstrations to NeuraCore.

```bash
python examples/2_collect_teleop_data_with_neuracore.py [--ip-address <quest-ip>] [--dataset-name <name>]
```

**Note**: You must be logged into NeuraCore.

**Arguments**:
- `--ip-address`: IP address of Meta Quest device (optional). Only needed when using WiFi connection. If not provided, defaults to auto-discovery via USB.
- `--dataset-name`: Name for the dataset (optional). If an existing dataset name is passed, the script will resume logging into this dataset. Otherwise, it will create a dataset with the specified name (or auto-generate one if not provided).

**Controls**:
- Same as script 1, plus:
- **Right Joystick Press**: Start/stop data recording

### 3. Replay NeuraCore Episodes

**Script**: `examples/3_replay_neuracore_episodes.py`

Replay recorded episodes from a NeuraCore dataset on the physical robot.

```bash
python examples/3_replay_neuracore_episodes.py --dataset-name <dataset-name> [--frequency <hz>] [--episode-index <index>]
```

**Arguments**:
- `--dataset-name`: Name of the NeuraCore dataset to replay
- `--frequency`: Playback frequency in Hz (default: 0). 0 plays the data aperiodically (not synchronized at a certain frequency as it was recorded). 
- `--episode-index`: Which episode to replay (default: 0). -1 will start replaying all the episodes one after the other.

**NOTE:** please be careful that the robot **will start moving** on the same trajectory that was recorded. Pressing `ctrl+C`
will gracefully disable the robot and it will cut power to the motors after 5 seconds.

### 4. Rollout NeuraCore Policy (Full GUI)

**Script**: `examples/4_rollout_neuracore_policy.py`

This script should be run after you have trained a policy and want to see it running on the robot. From a GUI, it allows you to:
- run the policy first and see the prediction horizon without moving the robot,
- then execute this prediction horizon, if wanted.
- or you can run and execute the prediction horizon immediately (for one prediction horizon only).
- or you can play the policy where prediction horizon will be generated and executed in a loop, until stopped

You also have all the same controls from **example_1**. This is to help you to manually move
the robot to a certain state and then run the policy for testing.


```bash
python examples/4_rollout_neuracore_policy.py --train-run-name <run-name> [--ip-address <quest-ip>]
```

or

```bash
python examples/4_rollout_neuracore_policy.py --model-path <path-to-model> [--ip-address <quest-ip>]
```

**Arguments**:
- `--train-run-name`: Name of the NeuraCore training run (fetches model from NeuraCore)
- `--model-path`: Local path to model file (alternative to train-run-name)
- `--ip-address`: IP address of Meta Quest device (optional). Only needed when using WiFi connection. If not provided, defaults to auto-discovery via USB.


### 5. Rollout NeuraCore Policy (Minimal)

**Script**: `examples/5_rollout_neuracore_policy_minimal.py`

Minimal version of policy rollout without GUI - This is a minimal clear example on how to deploy your policy with no extra features.

```bash
python examples/5_rollout_neuracore_policy_minimal.py --train-run-name <run-name>
```

or

```bash
python examples/5_rollout_neuracore_policy_minimal.py --model-path <path-to-model>
```

**Arguments**:
- `--train-run-name`: Name of the NeuraCore training run (fetches model from NeuraCore)
- `--model-path`: Local path to model file (alternative to train-run-name)

### 6. Visualize Policy from Dataset

**Script**: `examples/6_visualize_policy_from_dataset.py`

This script is useful when you don't have the robot ready but want to visualize how well the policy would perform. It visualizes policy predictions on episodes from a dataset without running on the robot. Useful for debugging and analysis.

```bash
python examples/6_visualize_policy_from_dataset.py --dataset-name <dataset-name> --train-run-name <run-name>
```

or

```bash
python examples/6_visualize_policy_from_dataset.py --dataset-name <dataset-name> --model-path <path-to-model>
```

**Arguments**:
- `--dataset-name`: Name of the NeuraCore dataset to visualize
- `--train-run-name`: Name of the NeuraCore training run (fetches model from NeuraCore)
- `--model-path`: Local path to model file (alternative to train-run-name)

## Configuration

Most configuration parameters are defined in `examples/common/configs.py`. Key parameters include:

- **IK Solver Parameters**: Position/orientation costs, damping, solver settings
- **Controller Filter**: One-euro filter parameters for smoothing controller input
- **Scaling Factors**: Translation and rotation scaling for teleoperation
- **Thread Rates**: Control loop frequencies for different components
- **Robot Parameters**: Neutral joint angles, joint names, frame names

## Project Structure

```
example_agilex/
├── examples/              # Example scripts
│   ├── 1_tune_teleop_params.py
│   ├── 2_collect_teleop_data_with_neuracore.py
│   ├── 3_replay_neuracore_episodes.py
│   ├── 4_rollout_neuracore_policy.py
│   ├── 5_rollout_neuracore_policy_minimal.py
│   ├── 6_visualize_policy_from_dataset.py
│   └── common/            # Shared utilities
│       ├── configs.py     # Configuration parameters
│       ├── data_manager.py
│       ├── policy_state.py
│       ├── robot_visualizer.py
│       └── threads/       # Background thread implementations
├── meta_quest_teleop/     # Meta Quest controller interface
├── piper_controller.py    # Robot controller interface
├── pink_ik_solver.py      # Inverse kinematics solver
├── piper_description/     # Robot URDF and meshes
└── environment.yaml       # Conda environment specification
```

## Troubleshooting

### Import Errors

- Ensure conda environment is activated: `conda activate piper-teleop`
- Verify all dependencies are installed: `conda env update -f environment.yaml`
- Check that `meta_quest_teleop` is installed: `pip install -e meta_quest_teleop/`

### Robot Communication Issues

- Verify CAN interface is active: `ip link show can0`
- You can activate CAN interface with: `bash scripts/piper/can_activate.sh can0 1000000`
- Check robot power and CAN bus connection
- Ensure robot is in the correct mode for control

### NeuraCore Connection Issues

- Verify you're logged in: `neuracore login`
- Check network connectivity to NeuraCore servers
- Verify dataset/run names are correct

## Safety Notes

⚠️ **IMPORTANT**: This software controls a physical robot. Always:

- Keep emergency stop accessible
- Start with robot disabled (Button A)
- Test in a safe area with no obstacles
- Monitor robot behavior closely, especially during first use
- Use the dead man's switch (grip button) - robot stops when released
- Ensure proper workspace clearance

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please follow the project's coding standards and submit pull requests for review.

## Support

For issues and questions, please open an issue on the repository or contact the maintainers.

