#!/bin/bash

# Allow root user to access the host X server
xhost +local:root

# Runs the docker container, mounting the host's X11 socket and the .Xauthority file
# to allow GUI applications to display on the host's X server.
# The --gpus all flag is used to enable GPU support in the container.
# The --device /dev/dri flag is used to enable GPU access for rendering.
# The --cap-add=SYS_ADMIN flag is used to add the SYS_ADMIN capability to the container,
# which is required for certain operations.
# The -v flag is used to mount the host's directories into the container.
# The -e flag is used to set environment variables in the container.
# The --rm flag is used to remove the container after it exits. (I.e., dooes not persist after exit)
# The -it flag is used to run the container in interactive mode with a pseudo-TTY.

# REQUIRES: canUsbKvaserTesting can be found in the home directory of the host machine.
docker run --rm -it \
  --gpus all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/canUsbKvaserTesting/linuxcan:/root/canUsbKvaserTesting/linuxcan \
  -v $HOME/.Xauthority:/root/.Xauthority:ro \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=/root/.Xauthority \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --device /dev/dri \
  --cap-add=SYS_ADMIN \
  ros2-humble-gpu

# If GPU is not found on Docker when trying OpenGL, try running the following on host: (you probably don't have the proper drivers installed)
#
# sudo apt install nvidia-container-toolkit
# sudo nvidia-ctk runtime configure --runtime=docker
# sudo systemctl restart docker
#
# Also, add the flags:
#
# -e NVIDIA_VISIBLE_DEVICES=all \
# -e NVIDIA_DRIVER_CAPABILITIES=all \ 
