from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='perceptions',
            namespace='perceptions',
            executable='yolov5_zed_node',
            name='yolov5_zed_node'
        ),
        Node(
            package='perceptions',
            namespace='perceptions',
            executable='yolov5_zed2_node',
            name='yolov5_zed2_node'
        ),
        # TODO: include Lidar Node once ready
    ])