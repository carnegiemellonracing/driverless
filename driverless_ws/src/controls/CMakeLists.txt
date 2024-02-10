cmake_minimum_required(VERSION 3.17)
project(controls CXX CUDA)

set(CUDA_ARCH 75)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(NOT CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 14)
endif()

#if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#  add_compile_options(-Wall -Wextra -Wpedantic)
#endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(interfaces REQUIRED)
find_package(glm REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/../)


set(LIB_SOURCES
        src/mppi/mppi.cu
        src/state/state_estimator.cu
        src/cuda_globals/cuda_globals.cu
)

add_library(controls_lib SHARED ${LIB_SOURCES})
ament_target_dependencies(controls_lib rclcpp std_msgs interfaces glm)

add_executable(controller src/nodes/controller.cpp)
target_link_libraries(controller controls_lib CUDA::curand)

set_target_properties(controls_lib controller PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CUDA_ARCH}
)

ament_target_dependencies(controller rclcpp interfaces)

ament_export_targets(controls_libExport HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp std_msgs interfaces glm)

install(
    TARGETS controls_lib
    EXPORT controls_libExport
    DESTINATION lib/${PROJECT_NAME}
)

install(
    TARGETS controller
    DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(controls_test tests/unit_tests.cpp)
  target_include_directories(controls_test PUBLIC
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
          $<INSTALL_INTERFACE:include>
  )
#  ament_target_dependencies(controls dependencies ...)
#  target_link_libraries(controls_test controls_lib)
  set_target_properties(controls_test PROPERTIES
          CUDA_SEPARABLE_COMPILATION ON
          CUDA_ARCHITECTURES ${CUDA_ARCH}
  )
endif()

ament_package()
