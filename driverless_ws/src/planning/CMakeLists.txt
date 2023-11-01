cmake_minimum_required(VERSION 3.5)
project(planning)

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
find_package(message_filters REQUIRED)
find_package(eufs_msgs REQUIRED)
find_package(GSL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(matplotplusplus)

# add_executable(slam planning_codebase/EKF_SLAM.cpp planning_codebase/new_slam_correct.cpp)
add_executable(slam planning_codebase/EKF_SLAM.cpp)
# target_link_libraries(slam_test rclcpp message_filters eufs_msgs GSL OpenCV)
ament_target_dependencies(slam rclcpp message_filters eufs_msgs GSL Eigen3 OpenCV)

install(TARGETS
  slam
  DESTINATION lib/${PROJECT_NAME})

ament_package()
