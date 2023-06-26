source ~/.bashrc
cd eufs
colcon build
. install/setup.bash
ros2 launch eufs_launcher eufs_launcher.launch.py