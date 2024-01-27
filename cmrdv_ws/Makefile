SHELL := /bin/bash
b: install/setup.bash
	colcon build --packages-select cmrdv_planning || source install/setup.bash
r: 
	ros2 run cmrdv_planning SLAM_test
