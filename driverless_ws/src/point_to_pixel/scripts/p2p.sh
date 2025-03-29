#!/bin/bash
# Get the installation directory
INSTALL_DIR=$(dirname $(dirname $(which ros2)))

# Determine the package share directory
SHARE_DIR="install/point_to_pixel/share/point_to_pixel"

# Path to the parameters file
PARAMS_FILE="${SHARE_DIR}/config/params.yaml"

# Check if params file exists
if [ ! -f "$PARAMS_FILE" ]; then
    echo "Error: Params file not found at $PARAMS_FILE"
    exit 1
fi

# Run the point_to_pixel node with parameters
ros2 run point_to_pixel point_to    _pixel --ros-args --params-file "$PARAMS_FILE" "$@"