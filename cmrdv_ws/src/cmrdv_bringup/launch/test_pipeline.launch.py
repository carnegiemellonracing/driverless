from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    # this is already in test_dim?
    heartbeat = Node(
            package='cmrdv_common',
            namespace='',
            executable='heartbeat',
        )
    dim = Node(
            package='cmrdv_common',
            namespace='',
            executable='dim',
        )

    ld.add_action(heartbeat)
    ld.add_action(dim)

    return ld
