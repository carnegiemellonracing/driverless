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
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
endif()

set(CMAKE_PREFIX_PATH "~/driverless/driverless_ws/install/interfaces/share/interfaces/cmake")

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
find_package(Threads REQUIRED)
find_package(message_filters REQUIRED)
# find_package(Math REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/../)
include_directories(${PROJECT_SOURCE_DIR}/extern/include)
include_directories($<INSTALL_INTERFACE:include>)


  # set to copy the states to host
if (DISPLAY)
add_compile_options(-DDISPLAY)
endif()

if (PARANOID)
  add_compile_options(-DPARANOID)
endif()

if (DATA)
  message("HELLO DATA IS ON")
  add_compile_options(-DDATA)
  add_executable(offline_controller src/data_generation/offline_controller.cpp)
  target_link_libraries(offline_controller controls_lib)
  ament_target_dependencies(offline_controller rclcpp interfaces builtin_interfaces)
  target_compile_definitions(offline_controller PRIVATE DATA)
  set_target_properties(offline_controller PROPERTIES
          CUDA_SEPARABLE_COMPILATION ON
          CUDA_ARCHITECTURES ${CUDA_ARCH}
  )
  install(
        TARGETS offline_controller
        DESTINATION lib/${PROJECT_NAME}
  )
endif()

if (USESYSID)
  add_compile_options(-DUSESYSID)
endif()

if (ROSBAG)
  add_compile_options(-DROSBAG)
endif()


# **** Controls Lib *****

file(GLOB TWO_TRACK_SRCS src/model/two_track/codegen/*.cpp)

set(LIB_SOURCES
        src/mppi/mppi.cu
        src/state/state_estimator.cu
        src/state/state_projector.cu
        src/cuda_globals/cuda_globals.cu
        extern/src/glad.c
        src/utils/gl_utils.cpp
        ${TWO_TRACK_SRCS}
        src/midline/svm.cpp
        src/midline/svm_conv.cpp
        src/midline/svm_conv_fast.cpp
        src/midline/svm_conv_strat3.cpp
        src/midline/svm_conv_fast_double_binsearch.cpp
        #src/midline/svm_conv_strat4.cpp
        src/midline/svm_conv_stratnaive.cpp
        src/midline/svm_conv_test.cpp
        src/midline/cones.cpp
        src/state/naive_state_tracker.cpp
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
          -g -G -ggdb
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

set(LINUXCAN_PATH $ENV{LINUXCAN_PATH})
# Check if the path is set
if(NOT LINUXCAN_PATH)
    message(FATAL_ERROR "Please set the LINUXCAN_PATH environment variable to /path/to/canUsbKvaserTesting/linuxcan.")
endif()

set(CANLIB_INCLUDE "${LINUXCAN_PATH}/include")
set(CANLIB_LIBRARIES "${LINUXCAN_PATH}/canlib/libcanlib.so")

# Create an imported target
add_library(can_lib SHARED IMPORTED)
set_target_properties(can_lib PROPERTIES
  IMPORTED_LOCATION "${CANLIB_LIBRARIES}"
)
target_include_directories(can_lib INTERFACE ${CANLIB_INCLUDE})

add_executable(controller src/nodes/controller.cpp src/can/cmr_can.c)

target_link_libraries(controller controls_lib Threads::Threads m can_lib)
ament_target_dependencies(controller rclcpp interfaces builtin_interfaces message_filters)

install(
        TARGETS controller
        DESTINATION lib/${PROJECT_NAME}
)

# ***** Dead Reckon Node *****
add_executable(dead_reckon 
        src/nodes/dead_reckon.cpp 
        src/can/cmr_can.c)



target_link_libraries(dead_reckon Threads::Threads m can_lib)
ament_target_dependencies(dead_reckon rclcpp interfaces builtin_interfaces)

install(
        TARGETS dead_reckon
        DESTINATION lib/${PROJECT_NAME}
)

# ***** Info Visualizer Node *****
add_executable(info_visualizer src/nodes/info_visualizer.cpp)

target_link_libraries(info_visualizer controls_lib)
ament_target_dependencies(info_visualizer rclcpp interfaces builtin_interfaces)

install(
        TARGETS info_visualizer
        DESTINATION lib/${PROJECT_NAME}
)


# ***** Shared options *****

set_target_properties(controls_lib controller dead_reckon PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CUDA_ARCH}
)


if(BUILD_TESTING)
  add_executable(controls_test_node tests/test_node.cpp)
  target_link_libraries(controls_test_node controls_lib GSL::gsl)

  add_executable(controls_echo_node tests/echo_node/echo_node.cpp)

  add_executable(model_wrapper tests/model/model_wrapper.cu)
  target_link_libraries(model_wrapper controls_lib)
  set_target_properties(model_wrapper PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})

  add_executable(null_gps_spoofer tests/null_gps_spoofer.cpp)
  target_link_libraries(null_gps_spoofer controls_lib)

  ament_target_dependencies(controls_echo_node rclcpp glm interfaces std_msgs)
  ament_target_dependencies(controls_test_node rclcpp glm interfaces std_msgs)
  ament_target_dependencies(null_gps_spoofer rclcpp glm interfaces std_msgs)

  install(
      TARGETS controls_test_node null_gps_spoofer controls_echo_node
      DESTINATION lib/${PROJECT_NAME}
  )
endif()

ament_package()
