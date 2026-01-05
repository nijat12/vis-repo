#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
PYTHON_CMD="python3"
MIN_PYTHON_VERSION="3.8"

# --- Helper Functions ---
command_exists() {
    command -v "$1" &> /dev/null
}

version_gt() {
    test "$(printf 
%s\n
' "$@" | sort -V | head -n 1)" != "$2"
}

# --- Main Script ---
echo "Starting project setup..."

# 1. Install System Dependencies
echo "Step 1/5: Installing system dependencies (requires sudo)..."
if command_exists apt-get; then
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0
else
    echo "Warning: 'apt-get' not found. Skipping system dependency installation."
    echo "Please ensure you have python3, pip, venv, and GUI libraries (like libGL) installed."
fi

# 2. Check for Python, Pip, and Venv
echo "Step 2/5: Verifying Python environment..."
if ! command_exists $PYTHON_CMD; then
    echo "Error: Python 3 is not installed or not in PATH. Please install it."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python version: $PYTHON_VERSION"

if ! version_gt $PYTHON_VERSION $MIN_PYTHON_VERSION; then
    echo "Warning: Python version is $PYTHON_VERSION, but > $MIN_PYTHON_VERSION is recommended."
fi

# Check for venv module
if ! $PYTHON_CMD -c "import venv" &> /dev/null; then
    echo "Error: The 'venv' module is not available for your Python installation."
    echo "On Debian/Ubuntu, you might need to run: sudo apt-get install python3-venv"
    exit 1
fi

# 3. Set up Virtual Environment
echo "Step 3/5: Setting up Python virtual environment in './.venv'..."
if [ ! -d ".venv" ]; then
    $PYTHON_CMD -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment for the rest of the script
source .venv/bin/activate

# 4. Install Python Packages
echo "Step 4/5: Installing Python packages from requirements.txt..."
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Skipping package installation."
fi

# 5. Create default runtime_config.json
echo "Step 5/5: Creating default runtime configuration..."
if [ ! -f "runtime_config.json" ]; then
    cat <<EOL > runtime_config.json
{
    "ENABLE_KILLSWITCH": false,
    "ENABLED_PIPELINES": [
        "baseline_base",
        "strategy_1",
        "strategy_3",
        "strategy_5",
        "strategy_8",
        "strategy_9",
        "strategy_10",
        "strategy_11",
        "strategy_12",
        "strategy_13"
    ]
}
EOL
    echo "Created runtime_config.json."
else
    echo "runtime_config.json already exists."
fi


echo ""
echo "✅ Setup Complete!"
echo "To activate the virtual environment in your shell, run:"
echo "source .venv/bin/activate"
