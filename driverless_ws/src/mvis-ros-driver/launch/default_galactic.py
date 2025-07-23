# =======================================
# Launch output converters for locally executed ECU package, basic configuration
# ---------------------------------------
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    config = os.path.join(get_package_share_directory('movia'), 'config', 'default.yaml')

    movia_node=Node(
        package = 'movia_driver',
        namespace = 'microvision',
        executable = 'movia',
        prefix=['xterm -e gdb -ex run --args'],
        output='screen',
        parameters = [config],
        arguments=['--ros-args', '--log-level', 'INFO']
    )
    ld.add_action(movia_node)
    
    return ld
