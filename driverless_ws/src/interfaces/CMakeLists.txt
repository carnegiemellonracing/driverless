cmake_minimum_required(VERSION 3.5)
project(interfaces)

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
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

# if(BUILD_TESTING)
#   find_package(ament_lint_auto REQUIRED)
#   # the following line skips the linter which checks for copyrights
#   # uncomment the line when a copyright and license is not present in all source files
#   #set(ament_cmake_copyright_FOUND TRUE)
#   # the following line skips cpplint (only works in a git repo)
#   # uncomment the line when this package is not in a git repo
#   #set(ament_cmake_cpplint_FOUND TRUE)
#   ament_lint_auto_find_test_dependencies()
# endif()


find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CarROT.msg"
  "msg/ConeList.msg"
  "msg/ConePositions.msg"
  "msg/DataFrame.msg"
  "msg/PPMConeArray.msg"
  "msg/PPMConePoints.msg"
  "msg/PairROT.msg"
  "msg/Points.msg"
  "msg/Spline.msg"
  "msg/SplineList.msg"
  "msg/SplineFrames.msg"
  "msg/ControlAction.msg"
  "msg/ConeArray.msg"
  "msg/ControlsState.msg"
  "msg/ControllerInfo.msg"
  "msg/ActuatorsInfo.msg"
  "msg/EndToEndDebug.msg"
  "msg/SLAMChunk.msg" # SLAM Additions
  "msg/SLAMData.msg"
  "msg/SLAMPose.msg"
  "msg/ConeArrayWithOdom.msg" # End SLAM Additions
  DEPENDENCIES geometry_msgs sensor_msgs std_msgs # Add packages that above messages depend on, in this case geometry_msgs for Sphere.msg
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()


