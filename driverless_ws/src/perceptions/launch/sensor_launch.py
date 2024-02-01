from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='perceptions',
            namespace='perceptions',
            executable='zed_node',
            name='zed_node'
        ),
        Node(
            package='perceptions',
            namespace='perceptions',
            executable='zed2_node',
            name='zed2_node'
        ),
        Node(
            package='hesai_ros_driver',
            namespace='hesai_ros_driver',
            executable='hesai_ros_driver_node',
            name='hesai_ros_driver_node'
        )
    ])