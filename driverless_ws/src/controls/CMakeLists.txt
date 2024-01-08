cmake_minimum_required(VERSION 3.5)
project(controls)

set(NO_CUDA FALSE)

# disable cuda compilation
#set(NO_CUDA TRUE)

if(NO_CUDA)
  add_compile_definitions(CONTROLS_NO_CUDA)
else()
  enable_language(CUDA)
endif()

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
find_package(planning REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/../)


set(LIB_SOURCES
        src/interface/interface.cpp
        src/interface/cpu_environment.cpp
        src/mppi/cpu_mppi.cpp)
if(NOT NO_CUDA)
  set(LIB_SOURCES ${LIB_SOURCES}
          src/interface/cuda_environment.cu
          src/mppi/cuda_mppi.cu)
endif()

add_library(controls_lib SHARED ${LIB_SOURCES})
ament_target_dependencies(controls_lib planning rclcpp std_msgs)

add_executable(controller src/nodes/controller.cpp)
target_link_libraries(controller controls_lib)

ament_target_dependencies(controller rclcpp)

ament_export_targets(controls_libExport HAS_LIBRARY_TARGET)
ament_export_dependencies(planning rclcpp std_msgs)

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
