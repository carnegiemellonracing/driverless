#!/bin/bash

is_ubuntu_version() {
    # Check if the OS is Linux
    if [[ "$(uname)" != "Linux" ]]; then
        echo ""
        return 1
    fi

    # Check if the architecture is 64-bit
    if [[ "$(uname -m)" != "x86_64" ]]; then
        echo ""
        return 1
    fi

    # Check if it's Ubuntu
    if ! grep -q 'Ubuntu' /etc/os-release; then
        echo ""
        return 1
    fi

    # Check Ubuntu version
    if grep -q 'VERSION="18.04' /etc/os-release; then
        echo "linux-64-ubuntu-bionic"
        return 0
    elif grep -q 'VERSION="20.04' /etc/os-release; then
        echo "linux-64-ubuntu-focal"
        return 0
    elif grep -q 'VERSION="22.04' /etc/os-release; then
        echo "linux-64-ubuntu-jammy"
        return 0
    elif grep -q 'VERSION="24.04' /etc/os-release; then
        echo "linux-64-ubuntu-noble"
        return 1
    else
        echo "linux-64-ubuntu-"
        return 1
    fi
}

ubuntu_version=$(is_ubuntu_version)
if [ -z "$ubuntu_version" ]; then
    echo "Unsupported linux platform ($ubuntu_version)!"
    exit 1
else
    echo "Found $ubuntu_version."
fi

check_ros1_installation() {
    # Check if ROS_DISTRO is set
    if [ -z "$ROS_DISTRO" ]; then
        echo "ROS_DISTRO is not set. ROS 1 might not be installed or sourced."
        return 1
    fi

    # Check if ROS_VERSION is set and equals 1
    if [ -z "$ROS_VERSION" ] || [ "$ROS_VERSION" != "1" ]; then
        # ROS_VERSION is not set to 1. ROS 1 might not be installed or sourced.
        return 1
    fi

    # Check if essential ROS 1 commands are available
    if ! command -v roscore &> /dev/null; then
        echo "roscore command not found. ROS 1 might not be installed or sourced."
        return 1
    fi

    if ! command -v catkin_make &> /dev/null; then
        echo "catkin_make command not found. Make sure catkin is installed."
        return 1
    fi

    echo "ROS 1 installation check passed."
    return 0
}

check_ros2_installation() {
    # Check if ROS_DISTRO is set
    if [ -z "$ROS_DISTRO" ]; then
        echo "ROS_DISTRO is not set. ROS 2 might not be installed or sourced."
        return 1
    fi

    # Check if ROS_VERSION is set and equals 2
    if [ -z "$ROS_VERSION" ] || [ "$ROS_VERSION" != "2" ]; then
        # ROS_VERSION is not set to 2. ROS 2 might not be installed or sourced.
        return 1
    fi

    # Check if essential ROS 2 commands are available
    if ! command -v ros2 &> /dev/null; then
        echo "ros2 command not found. ROS 2 might not be installed or sourced."
        return 1
    fi

    if ! command -v colcon &> /dev/null; then
        echo "colcon command not found. Make sure colcon is installed."
        return 1
    fi

    echo "ROS 2 installation check passed."
    return 0
}

relativePathToScript=$(dirname "$0")

if check_ros1_installation; then
    echo "ROS 1 ${ROS_DISTRO} is properly installed and sourced. Proceeding with catkin_make..."

    cd ./ros/
    catkin_make install --cmake-args -DLIBPCAP_PATH="$PWD"/../libs/$ubuntu_version/lib/ -DLIBJPEG_PATH="$PWD"/../libs/$ubuntu_version/lib/ -DROS_CMAKE_PREFIX_PATH="$PWD"/../libs/$ubuntu_version/share/ -DMVIS_SDK_PLUGINS_PATH="$PWD"/../libs/$ubuntu_version/lib/

elif check_ros2_installation; then
    echo "ROS 2 ${ROS_DISTRO} is properly installed and sourced. Proceeding with colcon build..."

    cd "$DRIVERLESS/driverless_ws" || exit 3
    colcon build --packages-up-to movia --cmake-args -DLIBPCAP_PATH="$PWD"/src/movia-ros-driver/libs/$ubuntu_version/lib/ -DLIBJPEG_PATH="$PWD"/src/mvis-ros-driver/libs/$ubuntu_version/lib/ -DROS2_CMAKE_PREFIX_PATH="$PWD"/src/mvis-ros-driver/libs/$ubuntu_version/share/ -DMVIS_SDK_PLUGINS_PATH="$PWD"/../libs/$ubuntu_version/lib/
#debug    colcon build  --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLIBPCAP_PATH="$PWD"/libs/$ubuntu_version/lib/ -DLIBJPEG_PATH="$PWD"/../libs/$ubuntu_version/lib/ -DROS2_CMAKE_PREFIX_PATH="$PWD"/../libs/$ubuntu_version/share/ -DMVIS_SDK_PLUGINS_PATH="$PWD"/../libs/$ubuntu_version/lib/
else
    echo "ROS 1 and 2 installation check failed. Please install or source ROS before building."
    exit 1
fi
