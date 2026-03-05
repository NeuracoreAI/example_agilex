# Foot Pedal Control System

The Foot Pedal system allows you to control the robot and data collection using a 3-pedal USB device (or any keyboard-mapped input).

## Actions
- **ACTIVATE**: Enables the robot and aligns targets to current joint positions.
- **HOME**: Moves the robot arm to its predefined home position.
- **RECORD**: Toggles Neuracore data collection (Start/Stop recording).

## Setup and Configuration

### Remapping Keys
To map or change which pedal triggers which action, run the configuration script:

```bash
./scripts/foot_pedal/run_pedal_config.sh
```

Follow the prompts in the terminal:
1. When prompted for 'ACTIVATE', press the first pedal.
2. When prompted for 'HOME', press the second pedal.
3. When prompted for 'RECORD', press the third pedal.

The configuration is saved to `~/.neuracore/foot_pedal.json`.

### Running the Demo
To start controlling the robot with the pedals, use the demo script:

```bash
python3 scripts/foot_pedal/pedal_demo.py
```

## Developer Usage

The pedal system is integrated into the Neuracore core library. You can use it in your own scripts as follows:

```python
from neuracore.core.input_devices.foot_pedal import FootPedal

pedal = FootPedal()
pedal.on("activate", your_activate_function)
pedal.on("home", your_home_function)
pedal.on("record", your_record_function)

pedal.start()
```

## Troubleshooting
- If pedals are not detected, ensure the terminal window has focus.
- The `readchar` library must be installed for key detection.
- If using a serial-based pedal, use `list_potential_ports()` from `foot_pedal.py` to identify the device path.
