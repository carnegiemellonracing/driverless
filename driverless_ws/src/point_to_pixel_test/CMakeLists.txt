cmake_minimum_required(VERSION 3.8)
project(point_to_pixel_test)

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
find_package(OpenCV REQUIRED)

# Add zed-open-capture cpp library path/to/zed_open_capture
# link_directories(~/CMR/zed-open-capture/build/zed_open_capture)

set(LIBRARY_PATH ~/Documents/driverless-packages/zed-open-capture/build)

include_directories(~/Documents/driverless-packages/zed-open-capture/include)
include_directories(~/Documents/driverless-packages/zed-open-capture/examples/include)

# Point to Pixel Executable
add_executable(point_to_pixel src/point_to_pixel.cpp)
ament_target_dependencies(point_to_pixel rclcpp interfaces geometry_msgs Eigen3 OpenCV)

target_include_directories(point_to_pixel PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(point_to_pixel PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

# Zed Node Executable
add_executable(zed_node src/zed.cpp)
ament_target_dependencies(zed_node rclcpp sensor_msgs Eigen3 OpenCV)

target_include_directories(zed_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(zed_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

target_link_libraries(zed_node
  ~/Documents/driverless-packages/zed-open-capture/build/libzed_open_capture.so)  # Link ZED SDK library

target_link_libraries(point_to_pixel
  ~/Documents/driverless-packages/zed-open-capture/build/libzed_open_capture.so)  # Link ZED SDK library
target_compile_definitions(point_to_pixel PUBLIC VIDEO_MOD_AVAILABLE)

target_compile_definitions(zed_node PUBLIC VIDEO_MOD_AVAILABLE)

install(TARGETS point_to_pixel zed_node
  DESTINATION lib/${PROJECT_NAME})

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
