cmake_minimum_required(VERSION 3.5)
project(cmrdv_node_utils)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rcutils REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(class_loader REQUIRED)
find_package(composition_interfaces REQUIRED)
find_package(rcpputils REQUIRED)
find_package(Boost REQUIRED)

set(dependencies
    rclcpp
    rclcpp_lifecycle
    rcutils
    rclcpp_components
    ament_index_cpp
    class_loader
    composition_interfaces
    rcpputils
    Boost
    lifecycle_msgs
    class_loader
    rcpputils
)

include_directories(include)

add_library( cmrdv_lifecycle_node 
  src/cmrdv_lifecycle_node.cpp
  src/lifecycle_component_wrapper.cpp
)
ament_target_dependencies(cmrdv_lifecycle_node rclcpp lifecycle_msgs rclcpp_components rclcpp_lifecycle)
# ament_export_targets(cmrdv_lifecycle_nodeTargets HAS_LIBRARY_TARGET)
# ament_export_dependencies(${dependencies})

add_executable(lifecycle_component_wrapper_st src/lifecycle_component_wrapper_st.cpp)
ament_target_dependencies(lifecycle_component_wrapper_st ${dependencies})
target_link_libraries(lifecycle_component_wrapper_st cmrdv_lifecycle_node)

add_executable(lifecycle_component_wrapper_mt src/lifecycle_component_wrapper_mt.cpp)
ament_target_dependencies(lifecycle_component_wrapper_mt ${dependencies})
target_link_libraries(lifecycle_component_wrapper_mt cmrdv_lifecycle_node)

# add_executable(lifecycle_test_node src/lifecycle_test_node.cpp)
# ament_target_dependencies(lifecycle_test_node ${dependencies})

add_library(lifecycle_test_node SHARED src/lifecycle_test_node.cpp)
ament_target_dependencies(lifecycle_test_node ${dependencies})
rclcpp_components_register_nodes(lifecycle_test_node "cmrdv_node_utils::MinimalPublisher")

# add_executable(lifecycle_component_wrapper src/lifecycle_component_wrapper.cpp)
# ament_target_dependencies(lifecycle_component_wrapper rclcpp ament_index_cpp class_loader rcpputils composition_interfaces rclcpp_lifecycle lifecycle_msgs)
  
install(
  TARGETS cmrdv_lifecycle_node
  EXPORT cmrdv_lifecycle_nodeTargets
  DESTINATION lib/${PROJECT_NAME}
)

install(
  TARGETS lifecycle_component_wrapper_st lifecycle_component_wrapper_mt # lifecycle_test_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  TARGETS lifecycle_test_node
  DESTINATION lib
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
