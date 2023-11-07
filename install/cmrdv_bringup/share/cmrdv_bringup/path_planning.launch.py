from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python import get_package_share_directory


#Midline, SLAM, optimizer
def generate_launch_description():
    ld = LaunchDescription()

    midline = Node(
        package='cmrdv_planning',
        executable='midpoint',
    )

    #optimizer = Node(
    #    package='cmrdv_planning',
    #    executable='optimizer',
    #)

    #slam = Node(
    #    package='cmrdv_planning',
    #    executable= 'SLAM',
    #)


    ld.add_action(midline)
    #ld.add_action(optimizer)
    #ld.add_action(slam)

    return ld
