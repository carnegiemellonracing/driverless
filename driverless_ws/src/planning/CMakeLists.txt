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
  
  # Set the path to Python.h

  # Add the directory containing Python.h to the include path
  # include_directories(/usr/include/python3.8)
  # include_directories(/usr/include/x86_64-linux-gnu/python3.8)
  # Find Python
  find_package(Python3 COMPONENTS Interpreter Development)

  if (NOT Python3_FOUND)
    message(FATAL_ERROR "Python 3 interpreter and development files not found")
  endif()
  include_directories(${Python3_INCLUDE_DIRS})


  add_executable(dynamic_spline src/planning_codebase/raceline/dsTest.cpp src/planning_codebase/raceline/dynamicSpline.cpp src/planning_codebase/midline/generator.cpp src/planning_codebase/raceline/raceline.cpp)
  ament_target_dependencies(dynamic_spline interfaces rclcpp std_msgs geometry_msgs Eigen3 eufs_msgs GSL)
  # Include Python library path
link_directories(/usr/lib/x86_64-linux-gnu/)

# Link against Python library
target_link_libraries(dynamic_spline python3.8)

# Include matplotlib-cpp library path (adjust path as necessary)
include_directories(/home/planning/driverless/driverless/driverless_ws/src/matplotlib-cpp)
  # add_library(planning_export src/planning_codebase/raceline/raceline.cpp)
  # ament_target_dependencies(planning_export interfaces rclcpp std_msgs geometry_msgs Eigen3 eufs_msgs GSL)

  # add_executable(midpoint_test src/nodes/midpoint.cpp src/planning_codebase/midline/generator.cpp src/planning_codebase/raceline/raceline.cpp)
  # ament_target_dependencies(midpoint_test interfaces rclcpp std_msgs geometry_msgs Eigen3 eufs_msgs GSL)

  # ament_export_targets(planning_exportTargets HAS_LIBRARY_TARGET)
  # ament_export_dependencies(interfaces rclcpp std_msgs geometry_msgs Eigen3 eufs_msgs GSL)

  install(
    # TARGETS midpoint_test
    TARGETS dynamic_spline
    DESTINATION lib/${PROJECT_NAME}
  )

  # install(
    # TARGETS planning_export
    # EXPORT planning_exportTargets
    # TARGETS dynamic_spline
    # EXPORT dynamic_splineTargets
    # DESTINATION lib/${PROJECT_NAME}
  # )

  ament_package()
