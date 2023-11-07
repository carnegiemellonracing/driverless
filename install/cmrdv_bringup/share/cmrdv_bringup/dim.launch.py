from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    dim_heartbeat = Node(
        package='cmrdv_common',
        executable='dim_heartbeat'
    )

    dim_request = Node(
        package='cmrdv_common',
        executable='dim_request'
    )

    ld.add_action(dim_heartbeat)
    ld.add_action(dim_request)

    return ld
