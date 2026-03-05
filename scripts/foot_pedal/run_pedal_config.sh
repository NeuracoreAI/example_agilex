#!/bin/bash
# Script to run foot pedal configuration

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." &> /dev/null && pwd )"

# Run the python config script
PYTHONPATH="$REPO_ROOT/neuracore:$PYTHONPATH" python3 "$REPO_ROOT/neuracore/neuracore/core/input_devices/pedal_config.py"
