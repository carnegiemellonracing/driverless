#!/bin/bash

cd ~/Documents/driverless/driverless_ws
source install/setup.bash && source /opt/ros/foxy/setup.bash
sleep 3

while true; do
python3 /home/chip/Documents/driverless/pulse_old.py
sleep 10
done    