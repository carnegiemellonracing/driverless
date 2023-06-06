from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    control_node = Node(
            package='cmrdv_controls',
            namespace='',
            executable='controller',
        )

    ld.add_action(control_node)

    return ld
