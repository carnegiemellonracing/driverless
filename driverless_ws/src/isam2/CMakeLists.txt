cmake_minimum_required(VERSION 3.5)
project(isam2)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Indicate CMAKE_BUILD_TYPE
if (CMAKE_BUILD_TYPE STREQUAL Debug)
  message("<---- Building in Debug mode ---->")
endif()
if (NOT CMAKE_BUILD_TYPE STREQUAL Debug)
  message("<---- Building NOT Debug mode ---->")
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  # remember that you may need to do -O0 when using gdb to debug
endif()

# find dependencies
find_package(geometry_msgs REQUIRED)
find_package(interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(message_filters REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)

# <----  Executables ---->
set(UTILS_SOURCES 
    src/utils/ros_utils.cpp
    src/utils/data_association.cpp
    src/utils/loop_closure.cpp
    )

set(ISAM2_SOURCES 
    src/isam2/isam2.cpp
    src/isam2/unary_factor.cpp
    src/isam2/slam_est_and_mcov.cpp
    )

set(GENERIC_SLAM_NODE_SOURCES
    src/nodes/generic_slam_node.cpp
    src/nodes/generic_slam_node_impl.cpp
    )

set(GENERAL_SOURCES 
    ${UTILS_SOURCES}
    ${ISAM2_SOURCES}
    ${GENERIC_SLAM_NODE_SOURCES}
    )
    
add_executable(real_data_slam_node_gps src/nodes/real_data_slam_node_gps.cpp ${GENERAL_SOURCES})
add_executable(real_data_slam_node_no_gps src/nodes/real_data_slam_node_no_gps.cpp ${GENERAL_SOURCES})
add_executable(controls_sim_slam_node src/nodes/controls_sim_slam_node.cpp ${GENERAL_SOURCES})

# Linking
if(CMAKE_BUILD_TYPE STREQUAL Debug)
  target_link_libraries(real_data_slam_node_gps PUBLIC libtbb.so libgtsamDebug.so Eigen3::Eigen libfmt.so)
  target_link_libraries(real_data_slam_node_no_gps PUBLIC libtbb.so libgtsamDebug.so Eigen3::Eigen libfmt.so)
  target_link_libraries(controls_sim_slam_node PUBLIC libtbb.so libgtsamDebug.so Eigen3::Eigen libfmt.so)
endif()
if(NOT CMAKE_BUILD_TYPE STREQUAL Debug)
  target_link_libraries(real_data_slam_node_gps PUBLIC libtbb.so libgtsam.so Eigen3::Eigen libfmt.so)
  target_link_libraries(real_data_slam_node_no_gps PUBLIC libtbb.so libgtsam.so Eigen3::Eigen libfmt.so)
  target_link_libraries(controls_sim_slam_node PUBLIC libtbb.so libgtsam.so Eigen3::Eigen libfmt.so)
endif()

# Linking with ROS dependencies
ament_target_dependencies(real_data_slam_node_gps PUBLIC rclcpp message_filters interfaces geometry_msgs sensor_msgs)
ament_target_dependencies(real_data_slam_node_no_gps PUBLIC rclcpp message_filters interfaces geometry_msgs sensor_msgs)
ament_target_dependencies(controls_sim_slam_node PUBLIC rclcpp message_filters interfaces geometry_msgs sensor_msgs)

# including dependencies
target_include_directories(real_data_slam_node_gps PUBLIC include ${GTSAM_INCLUDE_DIR} )
target_include_directories(real_data_slam_node_no_gps PUBLIC include ${GTSAM_INCLUDE_DIR} )
target_include_directories(controls_sim_slam_node PUBLIC include ${GTSAM_INCLUDE_DIR} )

install(TARGETS real_data_slam_node_gps real_data_slam_node_no_gps controls_sim_slam_node
        DESTINATION lib/${PROJECT_NAME})


# <---- isam2_lib ---->
add_library(isam2_lib SHARED src/nodes/real_data_slam_node_gps.cpp)
# target_link_libraries(isam2_lib PUBLIC gtsam)
if(CMAKE_BUILD_TYPE STREQUAL Debug)
  target_link_libraries(isam2_lib PUBLIC libtbb.so libgtsamDebug.so Eigen3::Eigen libfmt.so)
endif()
if(NOT CMAKE_BUILD_TYPE STREQUAL Debug)
  target_link_libraries(isam2_lib PUBLIC libtbb.so libgtsam.so Eigen3::Eigen libfmt.so)
endif()
ament_target_dependencies(isam2_lib PUBLIC rclcpp message_filters interfaces geometry_msgs sensor_msgs)
target_include_directories(isam2_lib PUBLIC include ${GTSAM_INCLUDE_DIR})
ament_export_include_directories(include)
ament_package()

