cmake_minimum_required(VERSION 3.12)
project(movia)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# if executed by cmake (building project sdk_ros)

# set(ROS2_CMAKE_PREFIX_PATH "" CACHE STRING "sdk_ros cmake prefix path")
# message(STATUS "ros2 CMAKE_PREFIX_PATH is: ${ROS2_CMAKE_PREFIX_PATH}")
# list(INSERT CMAKE_PREFIX_PATH 0 ${ROS2_CMAKE_PREFIX_PATH})

# if manually executed with colcon

execute_process(COMMAND lsb_release -cs OUTPUT_VARIABLE UBUNTU_CODENAME OUTPUT_STRIP_TRAILING_WHITESPACE)
set(UBUNTU_VERSION "linux-64-ubuntu-${UBUNTU_CODENAME}")
set(LIBS_BASE_PATH "${CMAKE_SOURCE_DIR}/libs/${UBUNTU_VERSION}") # Adjust if your 'libs' directory is elsewhere

set(ROS2_CMAKE_PREFIX_PATH "${LIBS_BASE_PATH}/share")
set(LIBPCAP_PATH "${LIBS_BASE_PATH}/lib")
set(LIBJPEG_PATH "${LIBS_BASE_PATH}/lib")
set(MVIS_SDK_PLUGINS_PATH "${LIBS_BASE_PATH}/lib")

message(STATUS "ROS2_CMAKE_PREFIX_PATH is: ${ROS2_CMAKE_PREFIX_PATH}")
message(STATUS "LIBPCAP_PATH is: ${LIBPCAP_PATH}")
message(STATUS "LIBJPEG_PATH is: ${LIBJPEG_PATH}")
message(STATUS "MVIS_SDK_PLUGINS_PATH is: ${MVIS_SDK_PLUGINS_PATH}")

# Add the ROS2_CMAKE_PREFIX_PATH to CMAKE_PREFIX_PATH
list(INSERT CMAKE_PREFIX_PATH 0 ${ROS2_CMAKE_PREFIX_PATH})

# --- HARDCODED PATHS END HERE ---


# MVIS_SDK_PATH should also be relative to CMAKE_SOURCE_DIR if it's within your project
# Assuming MVIS_SDK is directly in my_ros_workspace/libs/ubuntu_version/
set(MVIS_SDK_PATH "${LIBS_BASE_PATH}")


list(APPEND CMAKE_PREFIX_PATH "${MVIS_SDK_PATH}/target/")

message(STATUS "CMAKE_PREFIX_PATH is: ${CMAKE_PREFIX_PATH}")

# check if jpeg found
if(NOT EXISTS "${LIBJPEG_PATH}")
    message(FATAL_ERROR "No libjpeg found at: ${LIBJPEG_PATH} - please set LIBJPEG_PATH!")
endif()

# check if SDK plugins exist
if(NOT EXISTS "${MVIS_SDK_PLUGINS_PATH}")
    message(WARNING "No plugins for MVIS SDK found at: ${MVIS_SDK_PLUGINS_PATH} - please set MVIS_SDK_PLUGINS_PATH!")
endif()

# Set plugin path to load MOVIA plugins
add_compile_definitions(DEFAULT_PLUGIN_PATH="${CMAKE_INSTALL_PREFIX}/lib")

list(APPEND CMAKE_PREFIX_PATH ${MVIS_SDK_PATH}/sdk/${TARGET_INSTALL_PATH})
list(APPEND CMAKE_PREFIX_PATH ${MVIS_SDK_PATH}/logging/${TARGET_INSTALL_PATH})
list(APPEND CMAKE_PREFIX_PATH ${MVIS_SDK_PATH}/common/${TARGET_INSTALL_PATH})
list(APPEND CMAKE_PREFIX_PATH ${MVIS_SDK_PATH}/plugin_framework/${TARGET_INSTALL_PATH})
list(APPEND CMAKE_PREFIX_PATH ${TINYXML_PATH})

# find tinyxml2 and jpeg
if(WIN32)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_PREFIX_PATH}")
    set(TinyXML2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    set(tinyxml2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    set(JPEG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

    find_package(TinyXML2 REQUIRED)
    find_package(Boost REQUIRED COMPONENTS thread system program_options date_time filesystem)
    find_package(JPEG REQUIRED)

    message(STATUS "JPEG_FOUND: ${JPEG_FOUND}")
    message(STATUS "JPEG_LIBRARIES: ${JPEG_LIBRARIES}")
    message(STATUS "JPEG_INCLUDE_DIRS: ${JPEG_INCLUDE_DIRS}")
endif()

# find ROS2 dependencies
find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(image_transport REQUIRED)

# find sdk dependencies
find_package(microvision-common-logging REQUIRED)
find_package(microvision-common-plugin_framework REQUIRED)
find_package(microvision-common-sdk REQUIRED)

# Add the directory containing tinyxml2.lib to the library search path
link_directories(${TinyXML2_ROOT_DIR}/lib)

# executable

add_executable(movia src/movia.cpp)

target_include_directories(movia PRIVATE ${JPEG_INCLUDE_DIR} ${TinyXML2_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

set_target_properties(movia PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        VS_DEBUGGER_ENVIRONMENT "PATH=${COLLECTED_TARGET_BINARY_LOCATIONS};%PATH%"
        INSTALL_RPATH "$ORIGIN:$ORIGIN/."
        BUILD_WITH_INSTALL_RPATH TRUE
        SKIP_BUILD_RPATH FALSE
        INSTALL_RPATH_USE_LINK_PATH TRUE
)

target_compile_definitions(movia PUBLIC -DBOOST_ALL_NO_LIB -DBOOST_ALL_DYN_LINK -DBOOST_UUID_FORCE_AUTO_LINK -DBOOST_BIND_GLOBAL_PLACEHOLDERS)

ament_target_dependencies(movia
        rclcpp
        std_msgs
        sensor_msgs
        geometry_msgs
        tf2
        tf2_ros
        image_transport)

target_link_libraries(movia
        microvision-common::sdk
        microvision-common-plugin_framework::plugin_loader
        ${JPEG_LIBRARY}
        ${TinyXML2_LIBRARY})

install(TARGETS movia DESTINATION lib/${PROJECT_NAME})

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

# Add this before the copylibs() calls:
message(STATUS "DEBUG: MVIS_SDK_PATH = ${MVIS_SDK_PATH}")
message(STATUS "DEBUG: TARGET_INSTALL_PATH = ${TARGET_INSTALL_PATH}")
message(STATUS "DEBUG: LIBDIR = ${LIBDIR}")
message(STATUS "DEBUG: Copying from: ${MVIS_SDK_PATH}/sdk${TARGET_INSTALL_PATH}/${LIBDIR}")

# functionality to copy library files
macro(copylibs libpath)
    install(DIRECTORY "${libpath}/"
            DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
            FILES_MATCHING
            PATTERN "*.dll"
            PATTERN "*.pdb"
            PATTERN "*.so*"
            PATTERN "*SamplePlugin*" EXCLUDE
            PATTERN "CMakeFiles" EXCLUDE
            PATTERN "include" EXCLUDE
            PATTERN "tests" EXCLUDE
            PATTERN "share" EXCLUDE)
endmacro()

set(LIBDIR "lib")
if(WIN32)
    set(LIBDIR "bin")
endif()

# Copy required library files for SDK
copylibs("${MVIS_SDK_PATH}/sdk${TARGET_INSTALL_PATH}/${LIBDIR}")
copylibs("${MVIS_SDK_PATH}/logging${TARGET_INSTALL_PATH}/${LIBDIR}")
copylibs("${MVIS_SDK_PATH}/plugin_framework${TARGET_INSTALL_PATH}/${LIBDIR}")
copylibs("${MVIS_SDK_PATH}/plugin_framework${TARGET_INSTALL_PATH}/${LIBDIR}")

copylibs("${MVIS_SDK_PATH}/target/microvision-common-plugin_framework/plugin_loader")
copylibs("${MVIS_SDK_PATH}/target/tinyxml2")

# copy the MVIS SDK binary plugins to the install location
copylibs("${MVIS_SDK_PLUGINS_PATH}")

# Copy additional library files for SDK plugins
if(EXISTS "${LIBPCAP_PATH}")
    copylibs("${LIBPCAP_PATH}")
endif()
if(EXISTS "${LIBJPEG_PATH}")
    copylibs("${LIBJPEG_PATH}")
endif()

ament_package()
