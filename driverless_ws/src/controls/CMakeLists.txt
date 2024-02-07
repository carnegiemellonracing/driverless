cmake_minimum_required(VERSION 3.5)
project(controls)

enable_language(CUDA)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

#if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#  add_compile_options(-Wall -Wextra -Wpedantic)
#endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(interfaces REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/../)


set(LIB_SOURCES
        src/mppi/mppi.cu
        src/state/state_estimator.cu
        src/cuda_globals/cuda_globals.cu
)

add_library(controls_lib SHARED ${LIB_SOURCES})
ament_target_dependencies(controls_lib rclcpp std_msgs interfaces)

add_executable(controller src/nodes/controller.cpp)
target_link_libraries(controller controls_lib)

set_property(TARGET controls_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET controller PROPERTY CUDA_SEPARABLE_COMPILATION ON)

ament_target_dependencies(controller rclcpp interfaces)

ament_export_targets(controls_libExport HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp std_msgs interfaces)

install(
    TARGETS controls_lib
    EXPORT controls_libExport
    DESTINATION lib/${PROJECT_NAME}
)

install(
    TARGETS controller
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
