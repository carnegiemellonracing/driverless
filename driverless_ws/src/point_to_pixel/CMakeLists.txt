if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

cmake_minimum_required(VERSION 3.8)
project(point_to_pixel)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED) 
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

set(OpenCV_DIR /home/chip/Downloads/opencv/build/)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(/home/chip/Downloads/opencv/lib)

set(LIBRARY_PATH ~/Documents/driverless-packages/zed-open-capture/build)

include_directories(~/Documents/driverless-packages/zed-open-capture/include)
include_directories(~/Documents/driverless-packages/zed-open-capture/examples/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

set(CONE_HISTORY_NODE_SRC  
  src/transform/transform.cpp
  src/nodes/cone_history_node.cpp
  src/cones/cones.cpp
)

###############################

set(POINT_TO_PIXEL_SRC  
  src/transform/transform.cpp
  src/camera/camera.cpp
  lib/svm_lib.cpp
  src/cones/predictors/svm.cpp
  src/cones/cones.cpp
  src/cones/predictors/hsv.cpp
  src/cones/predictors/yolo.cpp
  src/nodes/point_to_pixel_node.cpp
)

# Cone History Test Executable
add_executable(cone_history_node ${CONE_HISTORY_NODE_SRC} )
ament_target_dependencies(cone_history_node rclcpp interfaces geometry_msgs Eigen3 OpenCV)
target_include_directories(cone_history_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(cone_history_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

# Point to Pixel Executable
add_executable(point_to_pixel ${POINT_TO_PIXEL_SRC})
ament_target_dependencies(point_to_pixel rclcpp interfaces geometry_msgs Eigen3 OpenCV)

target_include_directories(point_to_pixel PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(point_to_pixel PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

target_link_libraries(point_to_pixel
  ~/Documents/driverless-packages/zed-open-capture/build/libzed_open_capture.so  # Link ZED SDK library
  ${OpenCV_LIBS})  # Link OpenCV
target_compile_definitions(point_to_pixel PUBLIC VIDEO_MOD_AVAILABLE)

target_link_libraries(cone_history_node
  ~/Documents/driverless-packages/zed-open-capture/build/libzed_open_capture.so  # Link ZED SDK library
  ${OpenCV_LIBS})  # Link OpenCV
target_compile_definitions(cone_history_node PUBLIC VIDEO_MOD_AVAILABLE)

install(TARGETS point_to_pixel
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS cone_history_node
  DESTINATION lib/${PROJECT_NAME})


# Install wrapper script
install(
  FILES scripts/p2p.sh
  DESTINATION lib/${PROJECT_NAME}
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

# Install config files
install(
  DIRECTORY 
  config/
  DESTINATION share/${PROJECT_NAME}/config
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()