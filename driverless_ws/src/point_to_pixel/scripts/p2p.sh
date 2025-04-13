#!/bin/bash

# Path to the parameters file
PARAMS_FILE="$DRIVERLESS/driverless_ws/src/point_to_pixel/config/params.yaml"

# Check if params file exists
if [ ! -f "$PARAMS_FILE" ]; then
    echo "Error: Params file not found at $PARAMS_FILE"
    exit 1
fi

# Run the point_to_pixel node with parameters
ros2 run point_to_pixel point_to_pixel --ros-args --params-file "$PARAMS_FILE" "$@"