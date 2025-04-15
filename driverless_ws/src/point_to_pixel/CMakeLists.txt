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

# DEBUG OpenCV Version
# message(WARNING "OpenCV version: ${OpenCV_VERSION}")
# message(WARNING  "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
# message(WARNING "OpenCV libs: ${OpenCV_LIBS}")

# Add zed-open-capture cpp library path/to/zed_open_capture
# link_directories(~/CMR/zed-open-capture/build/zed_open_capture)

set(LIBRARY_PATH ~/Documents/driverless-packages/zed-open-capture/build)

include_directories(~/Documents/driverless-packages/zed-open-capture/include)
include_directories(~/Documents/driverless-packages/zed-open-capture/examples/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

set(POINT_TO_PIXEL_SRC  
  src/transform/transform.cpp
  src/camera/camera.cpp
  src/coloring/coloring.cpp
  src/coloring/hsv.cpp
  src/coloring/yolo.cpp
  src/ros_node/point_to_pixel_node.cpp
)

# Point to Pixel Executable
add_executable(point_to_pixel ${POINT_TO_PIXEL_SRC})
ament_target_dependencies(point_to_pixel rclcpp interfaces geometry_msgs Eigen3 OpenCV)

target_include_directories(point_to_pixel PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(point_to_pixel PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

# Point to Pixel Debug Executable
# add_executable(point_to_pixel_debug ${POINT_TO_PIXEL_SRC})
# ament_target_dependencies(point_to_pixel_debug rclcpp interfaces geometry_msgs Eigen3 OpenCV)

# target_include_directories(point_to_pixel_debug PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>)
# target_compile_features(point_to_pixel_debug PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

target_link_libraries(point_to_pixel
  ~/Documents/driverless-packages/zed-open-capture/build/libzed_open_capture.so  # Link ZED SDK library
  ${OpenCV_LIBS})  # Link OpenCV
target_compile_definitions(point_to_pixel PUBLIC VIDEO_MOD_AVAILABLE)

# target_link_libraries(point_to_pixel_debug
#   ~/Documents/driverless-packages/zed-open-capture/build/libzed_open_capture.so)  # Link ZED SDK library
# target_compile_definitions(point_to_pixel_debug PUBLIC VIDEO_MOD_AVAILABLE)

install(TARGETS point_to_pixel
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