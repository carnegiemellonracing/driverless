cmake_minimum_required(VERSION 3.5)
project(controls)

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
find_package(planning REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/../)

add_library(controls_lib SHARED
        src/interface/interface.cpp
        src/mppi/mppi.cpp
)

add_executable(controller
        src/nodes/controller.cpp
)
target_link_libraries(controller
        controls_lib
)

add_executable(controls_test
        tests/linkTest.cpp
)

ament_target_dependencies(controls_lib planning rclcpp std_msgs)
ament_target_dependencies(controller rclcpp)
ament_target_dependencies(controls_test planning rclcpp std_msgs)

install(
    TARGETS controls_test controls_lib controller
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
