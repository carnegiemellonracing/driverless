from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python import get_package_share_directory

import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

# IMPORTANT DISTINCTION: If loading launch file from a package that you have installed through apt-get or through source
# you need to write "launch/{filename}" in the os.path.join
# If package is already in your workspace (i.e. one of the cmrdv_ packages) you don't need to write "launch/" as I've done here

def generate_launch_description():
   
    ld = launch.LaunchDescription([
        ComposableNodeContainer(
            package='cmrdv_node_utils',
            name='wrapper_container',
            executable='lifecycle_component_wrapper_st',
            namespace="",
            composable_node_descriptions=[

                ComposableNode(
                    package='cmrdv_node_utils',
                    plugin='cmrdv_node_utils::MinimalPublisher',
                    name='minimal_publisher'
                )
            ]
        )
    ])

    return ld
