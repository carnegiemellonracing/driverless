cmake_minimum_required(VERSION 3.17)
project(controls CXX CUDA)

set(CUDA_ARCH 75)
set(CMAKE_INSTALL_RPATH "$ORIGIN")

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

if (PUBLISH_STATES)
  # set to publish state trajectories
  add_compile_options(-DPUBLISH_STATES -g -O0)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(interfaces REQUIRED)
find_package(glm REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)


include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/../)


set(LIB_SOURCES
        src/mppi/mppi.cu
        src/state/state_estimator.cu
        src/cuda_globals/cuda_globals.cu
)

add_library(controls_lib SHARED ${LIB_SOURCES})
ament_target_dependencies(controls_lib rclcpp std_msgs interfaces glm)
target_link_libraries(controls_lib CUDA::curand)

add_executable(controller src/nodes/controller.cpp)
target_link_libraries(controller controls_lib)

set_target_properties(controls_lib controller PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CUDA_ARCH}
)

ament_target_dependencies(controller rclcpp interfaces visualization_msgs)

ament_export_targets(controls_libExport HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp std_msgs interfaces glm visualization_msgs)

target_compile_options(controls_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        -g
        -G
        >)

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
#  find_package(ament_cmake_gtest REQUIRED)
#  ament_add_gtest(controls_unit_tests tests/unit_tests.cpp)
#  target_link_libraries(controls_unit_tests controls_lib)

  add_executable(controls_test_node tests/test_node.cpp)
  target_link_libraries(controls_test_node controls_lib)

  set_target_properties(controls_test_node PROPERTIES
          CUDA_SEPARABLE_COMPILATION ON
          CUDA_ARCHITECTURES ${CUDA_ARCH}
  )

  ament_target_dependencies(controls_test_node rclcpp glm interfaces std_msgs visualization_msgs)

  install(
          TARGETS controls_test_node
          DESTINATION lib/${PROJECT_NAME}
  )


  add_executable(model_wrapper tests/model/model_wrapper.cu)
  target_link_libraries(model_wrapper controls_lib)

endif()

ament_package()
