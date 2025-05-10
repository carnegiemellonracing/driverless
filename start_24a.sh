#!/bin/bash
source='source install/setup.bash && source /opt/ros/foxy/setup.bash'
cd='cd ~/Documents/driverless/driverless_ws'

tmux new-session -d -s pipeline # start new detached tmux session
tmux split-window -h -p 70 -t pipeline # Split the window into two horizontal panes
tmux split-window -v -t pipeline:0.0 # Split the left pane (pane 0) vertically
tmux split-window -v -t pipeline:0.2 # Split the right pane (pane 2) vertically
tmux split-window -h -t pipeline:0.2 # Split the top right pane (pane 2) horizontally

# IMU
tmux send-keys -t pipeline:0.0 "$cd"  C-m
tmux send-keys -t pipeline:0.0 "$source"  C-m
tmux send-keys -t pipeline:0.0 'source ~/Documents/driverless-packages/Xsens_MTi_ROS_Driver_and_Ntrip_Client/install/setup.bash'  C-m
tmux send-keys -t pipeline:0.0 'ros2 launch xsens_mti_ros2_driver xsens_mti_node.launch.py'  C-m
tmux send-keys -t pipeline:0.0 'ros2 launch xsens_mti_ros2_driver xsens_mti_node.launch.py'  C-m
tmux send-keys -t pipeline:0.0 'ros2 launch xsens_mti_ros2_driver xsens_mti_node.launch.py'  C-m
tmux send-keys -t pipeline:0.0 'ros2 launch xsens_mti_ros2_driver xsens_mti_node.launch.py'  C-m
tmux send-keys -t pipeline:0.0 'ros2 launch xsens_mti_ros2_driver xsens_mti_node.launch.py'  C-m
tmux send-keys -t pipeline:0.0 'ros2 launch xsens_mti_ros2_driver xsens_mti_node.launch.py'  C-m
# in case imu is buggy

# LIDAR
tmux send-keys -t pipeline:0.1 "$cd"  C-m
tmux send-keys -t pipeline:0.1 "$source"  C-m
tmux send-keys -t pipeline:0.1 'sudo ptpd -m -i eno1'  C-m
tmux send-keys -t pipeline:0.1 'chip22a'  C-m
tmux send-keys -t pipeline:0.1 'ros2 run hesai_ros_driver hesai_ros_driver_node'  C-m

# POINT TO PIXEL
tmux send-keys -t pipeline:0.2 "$cd"  C-m
tmux send-keys -t pipeline:0.2 "$source"  C-m
tmux send-keys -t pipeline:0.2 'ros2 run point_to_pixel p2p.sh'  C-m
tmux send-keys -t pipeline:0.2 'ros2 run point_to_pixel p2p.sh'  C-m
tmux send-keys -t pipeline:0.2 'ros2 run point_to_pixel p2p.sh'  C-m
tmux send-keys -t pipeline:0.2 'ros2 run point_to_pixel p2p.sh'  C-m
tmux send-keys -t pipeline:0.2 'ros2 run point_to_pixel p2p.sh'  C-m
tmux send-keys -t pipeline:0.2 'ros2 run point_to_pixel p2p.sh'  C-m
# in case cameras are buggy

# CONE HISTORY
tmux send-keys -t pipeline:0.3 "$cd"  C-m
tmux send-keys -t pipeline:0.3 "$source"  C-m
tmux send-keys -t pipeline:0.3 'ros2 run point_to_pixel cone_history_test_node'  C-m

# CONTROLLER
# Start mps daemon, idk what the command is rn
tmux send-keys -t pipeline:0.4 "$cd"  C-m
tmux send-keys -t pipeline:0.4 "$source"  C-m
tmux send-keys -t pipeline:0.4 '. ~/.bashrc'  C-m
tmux send-keys -t pipeline:0.4 'ros2 run controls controller'  C-m


# ---------------IMU--------------------------PERCEPTIONS-----------
# |                            0.0 |           0.2 |           0.3 |
# |                                |               |               |
# |         xsens_mti_node         |      p2p      |     cone      |
# |                                |               |    history    |
# |                                |               |               |
# |--------------LIDAR-------------|-----------CONTROLS------------|
# |                            0.1 |                           0.4 |
# |                                |                               |
# |      hesai_ros_driver_node     |          controller           |
# |                                |                               |
# |                                |                               |
# ------------------------------------------------------------------