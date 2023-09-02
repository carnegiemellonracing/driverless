cmake_minimum_required(VERSION 3.5)
project(planning)

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
find_package(rclcpp REQUIRED)
find_package(message_filters REQUIRED)
find_package(eufs_msgs REQUIRED)

add_executable(slam_test src/EKF_SLAM.cpp)
ament_target_dependencies(slam_test rclcpp message_filters eufs_msgs)

install(TARGETS
  slam_test
  DESTINATION lib/${PROJECT_NAME})

find_package(interfaces REQUIRED)                         # CHANGE
find_package(std_msgs REQUIRED)

# add_executable(talker src/publisher_member_function.cpp)
# ament_target_dependencies(talker rclcpp interfaces)         # CHANGE

# add_executable(listener src/subscriber_member_function.cpp)
# ament_target_dependencies(listener rclcpp interfaces)     # CHANGE

# install(TARGETS
#   talker
#   listener
#   DESTINATION lib/${PROJECT_NAME})

add_executable(planning planning_codebase/hello.cpp)
find_package(NumCpp 2.11.0 REQUIRED)
target_link_libraries(planning
       NumCpp::NumCpp)
ament_environment_hooks()
ament_package()
