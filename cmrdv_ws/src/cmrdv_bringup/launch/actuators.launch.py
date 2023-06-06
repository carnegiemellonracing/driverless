from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    fsm = Node(
            package='cmrdv_actuators',
            executable='fsm', # does this have any executables?
    )
    '''brake = Node(
            package='cmrdv_actuators',
            executable='brake', # does this have any executables?
    )
    '''
    steering = Node(
            package='cmrdv_actuators',
            executable='steering', # does this have any executables?
    )
    ld.add_action(fsm)
    # ld.add_action(brake)
    ld.add_action(steering)

    return ld
