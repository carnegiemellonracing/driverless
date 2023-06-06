from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    perceptions = Node(
        package='cmrdv_perceptions',
        executable='stereo_vision',
    )

    visualizer = Node(
        package='cmrdv_perceptions',
        executable='visualization'
    )

    ld.add_action(perceptions)
    # ld.add_action(visualizer)

    return ld
