cmake_minimum_required(VERSION 3.5)
project(planning)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(message_filters REQUIRED)
find_package(eufs_msgs REQUIRED)
find_package(Eigen3 3.3.7 REQUIRED)
link_directories(${Eigen_INCLUDE_DIRS})
find_package(GTSAM REQUIRED)

add_executable(slam_test planning_codebase/isamNode.cpp)
ament_target_dependencies(slam_test rclcpp GTSAM Eigen3 eufs_msgs)
ament_export_include_directories(source)

install(TARGETS
  slam_test
  DESTINATION lib/${PROJECT_NAME})

ament_environment_hooks()
ament_package()
