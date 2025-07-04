#!/bin/bash
source='source install/setup.bash && source /opt/ros/foxy/setup.bash'
cd_documents='cd ~/Documents/driverless/driverless_ws'
cd_downloads='cd ~/Downloads/driverless/driverless_ws'
cd_hesai='cd ~/Downloads/HesaiLidar_ROS_2.0/'

tmux new-session -d -s pipeline # start new detached tmux session
tmux split-window -h -p 70 -t pipeline # Split the window into two horizontal panes
tmux split-window -v -t pipeline:0.0 # Split the left pane (pane 0) vertically
tmux split-window -v -t pipeline:0.2 # Split the right pane (pane 2) vertically

# CONTROLLER
tmux send-keys -t pipeline:0.3 "nvidia-cuda-mps-control -d"  C-m
sleep 1
tmux send-keys -t pipeline:0.3 "$cd_documents"  C-m
tmux send-keys -t pipeline:0.3 "$source"  C-m
tmux send-keys -t pipeline:0.3 '. ~/.bashrc'  C-m
tmux send-keys -t pipeline:0.3 'ros2 run controls controller'  C-m

# END_TO_END_NODE
tmux send-keys -t pipeline:0.2 "$cd_downloads"  C-m
tmux send-keys -t pipeline:0.2 "$source"  C-m
tmux send-keys -t pipeline:0.2 'ros2 run perceptions endtoend_node'  C-m

# IMU
tmux send-keys -t pipeline:0.0 "$cd_documents"  C-m
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
tmux send-keys -t pipeline:0.1 "$cd_hesai"  C-m
tmux send-keys -t pipeline:0.1 "$source"  C-m
tmux send-keys -t pipeline:0.1 'sudo ptpd -m -i eno1'  C-m
tmux send-keys -t pipeline:0.1 'chip22a'  C-m
sleep 1
tmux send-keys -t pipeline:0.1 'ros2 run hesai_ros_driver hesai_ros_driver_node'  C-m

# ---------------IMU--------------------------PERCEPTIONS-----------
# |                            0.0 |                           0.2 |
# |                                |                               |
# |         xsens_mti_node         |         endtoend_node         |
# |                                |                               |
# |                                |                               |
# |--------------LIDAR-------------|-----------CONTROLS------------|
# |                            0.1 |                           0.3 |
# |                                |                               |
# |      hesai_ros_driver_node     |          controller           |
# |                                |                               |
# |                                |                               |
# ------------------------------------------------------------------