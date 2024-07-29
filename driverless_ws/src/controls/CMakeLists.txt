cmake_minimum_required(VERSION 3.16.3)
project(controls CXX CUDA C)

set(CUDA_ARCH 75)
set(CMAKE_INSTALL_RPATH "$ORIGIN")

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
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
        extern/src/glad.c
        src/utils/gl_utils.cpp
        ${TWO_TRACK_SRCS}
)

if (DISPLAY)
  set(LIB_SOURCES
          ${LIB_SOURCES}
          src/display/display.cpp)
endif()

add_library(controls_lib SHARED ${LIB_SOURCES})
ament_target_dependencies(controls_lib rclcpp std_msgs interfaces glm)
target_link_libraries(controls_lib CUDA::curand OpenGL::GL OpenGL::GLU ${SDL2_LIBRARIES})

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(controls_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
          -g --generate-line-info
          >)
else ()
  add_compile_options(-O3)
endif()

target_compile_options(controls_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math --expt-relaxed-constexpr
        >)

ament_export_targets(controls_libExport HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp std_msgs interfaces glm builtin_interfaces)

install(
        TARGETS controls_lib
        EXPORT controls_libExport
        DESTINATION lib/${PROJECT_NAME}
)



# ****** Controller *****

# add_executable(controller src/nodes/controller.cpp)
add_library(controller SHARED
  src/nodes/controller.cpp
)
# target_link_libraries(controller controls_lib)
ament_target_dependencies(controller rclcpp rclcpp_components interfaces builtin_interfaces)
rclcpp_components_register_nodes(controller "controls::nodes::ControllerNode")

install(
  TARGETS controller
  DESTINATION lib/${PROJECT_NAME}
)


# install(
#         TARGETS controller
#         DESTINATION lib/${PROJECT_NAME}
# )


# ****** Controller Component *****

# add_executable(controller_component
#   src/nodes/controller_component.cpp
# )
add_library(controller_component SHARED
  src/nodes/controller_component.cpp
)
ament_target_dependencies(controller_component rclcpp rclcpp_components std_msgs interfaces)
rclcpp_components_register_nodes(controller_component "controller::ControllerComponent")

# add_executable(controller_component_st
#   src/nodes/controller_component_st.cpp
# )
# ament_target_dependencies(controller_component_st rclcpp rclcpp_components std_msgs interfaces)

# install(
#       TARGETS controller_component controller_component_st
#       DESTINATION lib/${PROJECT_NAME}
#   )

# add_executable(controller_component_mt
#   src/nodes/controller_component_mt.cpp
# )
# ament_target_dependencies(controller_component_mt rclcpp std_msgs interfaces)

# install(DIRECTORY
#   launch
#   DESTINATION share/${PROJECT_NAME}
# )

add_executable(controller_executor
  src/nodes/controller_executor.cpp
)
target_link_libraries(controller_executor controller controls_lib)
ament_target_dependencies(controller_executor rclcpp rclcpp_components interfaces builtin_interfaces)

install (
  TARGETS controller_executor
  DESTINATION lib/${PROJECT_NAME}
)


# install (TARGETS controller_component
#   ARCHIVE DESTINATION lib
#   LIBRARY DESTINATION lib
#   RUNTIME DESTINATION bin)

# ***** Shared options *****

set_target_properties(controls_lib controller PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CUDA_ARCH}
)

if(BUILD_TESTING)
  add_executable(controls_test_node tests/test_node.cpp)
  target_link_libraries(controls_test_node controls_lib GSL::gsl)

  add_executable(model_wrapper tests/model/model_wrapper.cu)
  target_link_libraries(model_wrapper controls_lib)

  add_executable(null_gps_spoofer tests/null_gps_spoofer.cpp)
  target_link_libraries(null_gps_spoofer controls_lib)

  ament_target_dependencies(controls_test_node rclcpp glm interfaces std_msgs)
  ament_target_dependencies(null_gps_spoofer rclcpp glm interfaces std_msgs)

  install(
      TARGETS controls_test_node null_gps_spoofer
      DESTINATION lib/${PROJECT_NAME}
  )
endif()

ament_package()
