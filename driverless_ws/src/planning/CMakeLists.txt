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
find_package(interfaces REQUIRED)


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
# include_directories(${PWD}/../interfaces)


#add_executable(slam_test src/planning_codebase/EKF_SLAM.cpp)
#ament_target_dependencies(slam_test rclcpp message_filters eufs_msgs)
#ament_export_include_directories(source)

# Temporary fix to compile raceline
add_library(frenet_test src/planning_codebase/raceline/frenet.cpp src/planning_codebase/raceline/raceline.cpp)
ament_target_dependencies(frenet_test rclcpp)
ament_export_include_directories(src)

add_library(midline src/midpoint.cpp src/planning_codebase/raceline/generator.cpp)
ament_target_dependencies(midline rclcpp interfaces)
ament_export_include_directories(src)

target_link_libraries(midline GSL::gsl GSL::gslcblas)
target_link_libraries(frenet_test GSL::gsl GSL::gslcblas)

#install(TARGETS
#  slam_test
#  DESTINATION lib/${PROJECT_NAME})

target_include_directories(midline PRIVATE AFTER ${PROJECT_SOURCE_DIR}/..)

install(TARGETS
	frenet_test
	DESTINATION lib/${PROJECT_NAME})


install(TARGETS
  midline
	DESTINATION lib/${PROJECT_NAME})

ament_environment_hooks()
ament_package()
