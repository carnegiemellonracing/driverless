from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cmrdv_common',
            namespace='',
            executable='heartbeat',
            name='dim'
        ),
        Node(
            package='cmrdv_common',
            namespace='',
            executable='dim',
            name='dim'
        )
    ])