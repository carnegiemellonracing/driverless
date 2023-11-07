from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python import get_package_share_directory

# IMPORTANT DISTINCTION: If loading launch file from a package that you have installed through apt-get or through source
# you need to write "launch/{filename}" in the os.path.join
# If package is already in your workspace (i.e. one of the cmrdv_ packages) you don't need to write "launch/" as I've done here

def generate_launch_description():
    ld = LaunchDescription()

    data_collection = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('cmrdv_bringup'),
                         'data.launch.py')
        )
    )

    perceptions = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('cmrdv_bringup'),
                         'perceptions.launch.py')
        )
    ) 

    path_planning = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('cmrdv_bringup'),
                         'path_planning.launch.py')
        )
    )

    ld.add_action(data_collection)
    ld.add_action(perceptions)
    ld.add_action(path_planning)

    return ld
