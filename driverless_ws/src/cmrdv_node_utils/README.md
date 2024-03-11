# cmrdv_node_utils

TODO: create package description.

## Composable Node Launch File Templates

As part of the changes outlined in the the saftey archeticture proposal, every subteam is expected to convert non-CMRDVLifecycle nodes into composable nodes in order to adhere to the Heartbeat System. 


### Non-CMRDVLifecycle Node Launch File Example

The following example shows how to wrap and launch non-CMRDVLifecycle node(s) (it then becomes a CMRDVLifecycle node) for testing.

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
   
    ld = launch.LaunchDescription([
        ComposableNodeContainer(
            package='cmrdv_node_utils',
            name='wrapper_container',
            executable='lifecycle_component_wrapper_st',
            namespace="",
            composable_node_descriptions=[
                # This non-lifecycle node is now a exposed as a lifecycle node
                ComposableNode(
                    package='cmrdv_node_utils',
                    plugin='cmrdv_node_utils::MinimalPublisher',
                    name='minimal_publisher'
                )
            ]
        )
    ])

    return ld
```


#### Details

Container
- ```package='cmrdv_node_utils'``` (REQUIRED) is the wrappers package name.

- ```name='wrapper_container''``` (REQUIRED) is the wrappers name. The actual name can be modified.

- ```executable='lifecycle_component_wrapper_st'``` (REQUIRED) is the wrapper executable.
    + You can choose to use:
        - ```'lifecycle_component_wrapper_st'``` Runs all nodes within the container on a single thread. Each node executed one after the other.
        - ```'lifecycle_component_wrapper_mt'``` Runs all nodes within the container on seperate threads. Nodes executed concurrently.

- ```namespace=""``` (OPTIONAL) is the namespace of the wrapper executable. Used to organize nodes. If you have multiple containers, you can assign namespace based on what each one does.

- ```composable_node_descriptions=[]``` (REQUIRED) stores non-CMRDVLifecycle nodes

Node

- ```package='cmrdv_node_utils'``` Same as above

- ```name='minimal_publisher'``` Same as above but OPTIONAL. If using multiple nodes with the same type (plugin), assign them unique names (e.g. "minimal_publisher_2').

- ```plugin='cmrdv_node_utils::MinimalPublisher'``` (REQUIRED) the source file for the node.


### CMRDVLifecycle Node Launch File Example
The following example shows how to launch CMRDVLifecycle node(s) for testing.


```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    ld = LaunchDescription([
        Node(
            package='<Node package name>',
            executable='<node executable (source code)>',
            name='<Node name>'
        ),
    ])

    return ld

```