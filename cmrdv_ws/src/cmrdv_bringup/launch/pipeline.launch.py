import os
import sys

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python import get_package_share_directory


def generate_launch_description():
    trackdrive = False

    for arg in sys.argv:
        if arg.startswith("mode:="):
            mode = str(arg.split(":=")[1])
            if mode == "trackdrive": trackdrive = True

    ld = LaunchDescription()

    heartbeat = Node (
        package='cmrdv_common',
        executable='heartbeat'
    )

    actuators = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('cmrdv_bringup'),
                        'actuators.launch.py')
        )
    )
    # ld.add_action(actuators)
    ld.add_action(heartbeat)

    if trackdrive:

        # data_collection = IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(
        #         os.path.join(get_package_share_directory('cmrdv_bringup'),
        #                     'data.launch.py')
        #     )
        # )
        dim = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('cmrdv_bringup'),
                            'dim.launch.py')
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

        controls = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('cmrdv_bringup'),
                            'controls.launch.py')
            )
        )

        # ld.add_action(data_collection)
        ld.add_action(dim)
        ld.add_action(perceptions)
        ld.add_action(path_planning)
        ld.add_action(controls)

    else: 
        controls_auto = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('cmrdv_bringup'),
                            'controls_auto.launch.py')
            )
        )
        ld.add_action(controls_auto)

    return ld
