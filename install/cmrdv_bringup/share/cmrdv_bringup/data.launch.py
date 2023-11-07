from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python import get_package_share_directory

def generate_launch_description():
    ld = LaunchDescription()

    data_collection = Node(
        package='cmrdv_collection',
        executable='data_collection',
    )

    #velodyne_launch = IncludeLaunchDescription(
    #    PythonLaunchDescriptionSource(
    #        os.path.join(get_package_share_directory('velodyne'),
    #                     'launch/velodyne-all-nodes-VLP16-launch.py')
    #    )
    #) 

    # zed_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(get_package_share_directory('zed_wrapper'),
    #                      'launch/zed2.launch.py')
    #     )
    # )

    # sbg_launch = IncludeLaunchDescription(
    #    PythonLaunchDescriptionSource(
    #       os.path.join(get_package_share_directory('sbg_driver'),
    #                    'launch/sbg_device_launch.py')
    #   )
    #
    ld.add_action(data_collection)
    #ld.add_action(velodyne_launch)
    # ld.add_action(zed_launch)
    #ld.add_action(sbg_launch)

    return ld
