cmake_minimum_required(VERSION 3.17)
project(controls CXX CUDA C)

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

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(interfaces REQUIRED)
find_package(glm REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GSL REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/../)
include_directories(${PROJECT_SOURCE_DIR}/extern/include)
include_directories($<INSTALL_INTERFACE:include>)

if (DISPLAY)
  # set to copy the states to host
  add_compile_options(-DDISPLAY)
endif()

if (PARANOID)
  add_compile_options(-DPARANOID)
endif()


# **** Controls Lib *****

file(GLOB TWO_TRACK_SRCS src/model/two_track/codegen/*.cpp)

set(LIB_SOURCES
        src/mppi/mppi.cu
        src/state/state_estimator.cu
        src/cuda_globals/cuda_globals.cu
        src/display/display.cpp
        extern/src/glad.c
        src/utils/gl_utils.cpp
        ${TWO_TRACK_SRCS}
)

add_library(controls_lib SHARED ${LIB_SOURCES})
ament_target_dependencies(controls_lib rclcpp std_msgs interfaces glm)
target_link_libraries(controls_lib CUDA::curand OpenGL::GL OpenGL::GLU ${SDL2_LIBRARIES})

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(controls_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
          -g --generate-line-info --use_fast_math
          >)
endif()

ament_export_targets(controls_libExport HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp std_msgs interfaces glm visualization_msgs)

install(
        TARGETS controls_lib
        EXPORT controls_libExport
        DESTINATION lib/${PROJECT_NAME}
)



# ****** Controller *****

add_executable(controller src/nodes/controller.cpp)
target_link_libraries(controller controls_lib)
ament_target_dependencies(controller rclcpp interfaces visualization_msgs)

install(
        TARGETS controller
        DESTINATION lib/${PROJECT_NAME}
)


# ***** Shared options *****

set_target_properties(controls_lib controller PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CUDA_ARCH}
)


if(BUILD_TESTING)
  add_executable(controls_test_node tests/test_node.cpp)
  target_link_libraries(controls_test_node controls_lib GSL::gsl)

  ament_target_dependencies(controls_test_node rclcpp glm interfaces std_msgs)

  install(
      TARGETS controls_test_node
      DESTINATION lib/${PROJECT_NAME}
  )
endif()

ament_package()
