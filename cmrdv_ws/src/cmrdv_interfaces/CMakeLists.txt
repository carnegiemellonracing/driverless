cmake_minimum_required(VERSION 3.8)
project(cmrdv_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(eufs_msgs REQUIRED)
find_package(sbg_driver REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# List of message files
set(msg_files
  "msg/DataFrame.msg"
  "msg/SimDataFrame.msg"
  "msg/ConeList.msg"
  "msg/VehicleState.msg"
  "msg/CarROT.msg"
  "msg/ControlAction.msg"
  "msg/ConePositions.msg"
  "msg/Heartbeat.msg"
  "msg/Points.msg"
  "msg/Brakes.msg"
  "msg/PairROT.msg"
)

# Generate messages & services
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${action_files}
  ${srv_files}
  DEPENDENCIES eufs_msgs std_msgs sbg_driver
)

ament_export_dependencies(rosidl_default_runtime)
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
