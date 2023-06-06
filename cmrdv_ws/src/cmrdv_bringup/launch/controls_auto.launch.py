from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    controls_auto = Node(
            package='cmrdv_controls',
            executable='autotest_controller',
        )
    ld.add_action(controls_auto)

    return ld
