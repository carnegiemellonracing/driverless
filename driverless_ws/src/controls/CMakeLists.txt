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

add_executable(controls_test linkTest.cpp)
target_include_directories(controls_test PRIVATE ${PROJECT_SOURCE_DIR}/../)

ament_target_dependencies(controls_test planning rclcpp std_msgs)
# message(WARNING ${PROJECT_SOURCE_DIR})

install(TARGETS
controls_test
DESTINATION lib/${PROJECT_NAME})

ament_package()
