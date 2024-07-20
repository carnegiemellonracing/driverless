import launch
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode



def generate_launch_description():
    container = ComposableNodeContainer(
        name='controls_container',
        namespace='',
        package='controls',
        executable='controller_component_st',
        composable_node_descriptions=[
            ComposableNode(
                package='controls',
                plugin='controls::ControllerComponent',
                name='controller_component'
            ),
            ComposableNode(
                package='controls',
                plugin='controls::ControllerComponent',
                name='controller_component2'
            ),
        ],
        output='screen',
    )

    return LaunchDescription([container])

if __name__ == '__main__':
    generate_launch_description()