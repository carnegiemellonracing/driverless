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

#add_executable(slam_test src/planning_codebase/EKF_SLAM.cpp)
#ament_target_dependencies(slam_test rclcpp message_filters eufs_msgs)
#ament_export_include_directories(source)

# Temporary fix to compile raceline
add_library(frenet_test src/raceline/frenet.cpp src/raceline/raceline.cpp)
ament_target_dependencies(frenet_test rclcpp)
ament_export_include_directories(src)

target_link_libraries(frenet_test GSL::gsl GSL::gslcblas)

#install(TARGETS
#  slam_test
#  DESTINATION lib/${PROJECT_NAME})

install(TARGETS
	frenet_test
	DESTINATION lib/${PROJECT_NAME})

ament_environment_hooks()
ament_package()
