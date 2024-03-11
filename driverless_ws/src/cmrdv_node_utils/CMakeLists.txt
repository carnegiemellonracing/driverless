cmake_minimum_required(VERSION 3.5)
project(cmrdv_node_utils)

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
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Includes
include_directories(
  include
)

# Build
ament_auto_add_library(${PROJECT_NAME} SHARED
  src/cmrdv_lifecycle_node.cpp
)

ament_auto_add_library(lifecycle_component_wrapper SHARED
  src/lifecycle_component_wrapper.cpp
)

ament_auto_add_executable(lifecycle_component_wrapper_st
  src/lifecycle_component_wrapper_st.cpp
)

ament_auto_add_executable(lifecycle_component_wrapper_mt
  src/lifecycle_component_wrapper_mt.cpp
)


# Link local targets
target_link_libraries(lifecycle_component_wrapper
  ${PROJECT_NAME}
)

target_link_libraries(lifecycle_component_wrapper_st
  lifecycle_component_wrapper
  ${PROJECT_NAME}
)

target_link_libraries(lifecycle_component_wrapper_mt
  lifecycle_component_wrapper
  ${PROJECT_NAME}
)



# REMOVE ONCE TESTING IS DONE
ament_auto_add_library(minimalPublisher SHARED
  src/minimalPublisher.cpp
)
target_link_libraries(minimalPublisher
  ${PROJECT_NAME}
)
rclcpp_components_register_nodes(minimalPublisher "cmrdv_node_utils::MinimalPublisher")


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()
