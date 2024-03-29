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
    
    # stereo_mode = Node(
    #     package='stereo',
    #     executable='stereo_cones'
    # )

    # lidar_node = Node(
    #     package='perceptions',
    #     executable='lidar_sub'
    # )

    velodyne = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('velodyne'),
                         'velodyne-all-nodes-VLP16-launch.py')
        )
    )

    # sbg = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(get_package_share_directory('sbg_driver'),
    #                      'sbg_driver_launch.py')
    #     )
    # )
    # zed = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(get_package_share_directory('zed_wrapper'),
    #                      'zed2.launch.py')
    #     )
    # )

    # ld.add_action(stereo_mode)
    # ld.add_action(lidar_node)
    ld.add_action(velodyne)
    # ld.add_action(sbg)

    return ld
