  cmake_minimum_required(VERSION 3.5)
  project(planning)

  # Default to C++14
  if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
  endif()

  if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O0)
  endif()

  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(eufs_msgs REQUIRED)
  find_package(Eigen3 3.3 REQUIRED)
  find_package(GSL REQUIRED)
  find_package(interfaces REQUIRED)

  add_library(planning_export src/planning_codebase/raceline/raceline.cpp)
  ament_target_dependencies(planning_export interfaces rclcpp std_msgs geometry_msgs Eigen3 eufs_msgs GSL)


  add_executable(midpoint_test src/nodes/midpoint.cpp src/planning_codebase/midline/generator.cpp src/planning_codebase/raceline/raceline.cpp)
  # add_executable(midline_test src/generator.cpp src/raceline/raceline.cpp)
  ament_target_dependencies(midpoint_test interfaces rclcpp std_msgs geometry_msgs Eigen3 eufs_msgs GSL)

  ament_export_targets(planning_exportTargets HAS_LIBRARY_TARGET)
  ament_export_dependencies(interfaces rclcpp std_msgs geometry_msgs Eigen3 eufs_msgs GSL)
  
  # ament_target_dependencies(midline_test rclcpp std_msgs geometry_msgs eufs_msgs GSL)

  install(
    TARGETS midpoint_test
    DESTINATION lib/${PROJECT_NAME}
  )

  install(
    TARGETS planning_export
    EXPORT planning_exportTargets
    DESTINATION lib/${PROJECT_NAME}
  )

  ament_package()
