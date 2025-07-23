@echo off
setlocal enabledelayedexpansion

:: Check if ROS_DISTRO is set
if not defined ROS_DISTRO (
    echo ROS_DISTRO is not set. ROS might not be installed or sourced.
    exit /b 1
)

:: Check if ROS_VERSION is set
if not defined ROS_VERSION (
    echo ROS_VERSION is not set. ROS might not be installed or sourced.
    exit /b 1
)

:: Check for common Visual Studio environment variables
if not defined VSINSTALLDIR if not defined VisualStudioVersion (
    echo Please use in an MSVC native command window.
    exit /b 2
)

echo Visual Studio version: !VisualStudioVersion!

:: Check if Boost is installed
call :check_boost
if %errorlevel% neq 0 (
    echo Boost is not found. Please make sure Boost is installed and BOOST_ROOT is set.
    exit /b 1
)

:: Check ROS version
if "%ROS_VERSION%"=="1" (
    call :check_ros1
    
    echo ROS 1 %ROS_DISTRO% is properly installed and sourced. Proceeding with catkin_make ...

    cd ./ros/
    catkin_make install --cmake-args -DLIBPCAP_PATH="%CD:\=/%/libs/windows-64-vc16/lib/" -DLIBJPEG_PATH="%CD:\=/%/libs/windows-64-vc16/lib/" -DROS_CMAKE_PREFIX_PATH="%CD:\=/%/libs/windows-64-vc16/share/" -DMVIS_SDK_PLUGINS_PATH="%CD:\=/%/libs/windows-64-vc16/bin/"
    
) else if "%ROS_VERSION%"=="2" (
    call :check_ros2

    echo ROS 2 %ROS_DISTRO% is properly installed and sourced. Proceeding with colcon build ...

    cd ./ros2/
    colcon build --cmake-args -DTinyXML2_INCLUDE_DIRS="%CD:\=/%/libs/windows-64-vc16/include/tinyxml2" -DTinyXML2_LIBRARIES="%CD:\=/%/libs/windows-64-vc16/lib/tinyxml2.lib" -DLIBPCAP_PATH="%CD:\=/%/libs/windows-64-vc16/lib/" -DLIBJPEG_PATH="%CD:\=/%/libs/windows-64-vc16/lib/" -DROS2_CMAKE_PREFIX_PATH="%CD:\=/%/libs/windows-64-vc16/share/" -DMVIS_SDK_PLUGINS_PATH="%CD:\=/%/libs/windows-64-vc16/bin/"
	
) else (
    echo Unknown ROS_VERSION: %ROS_VERSION%
    exit /b 1
)

exit /b 0

:check_ros1
    :: Check for roscore command
    where roscore >nul 2>nul
    if %errorlevel% neq 0 (
        echo roscore command not found. ROS 1 might not be installed or sourced.
        exit /b 1
    )
    :: Check for catkin_make command
    where catkin_make >nul 2>nul
    if %errorlevel% neq 0 (
        echo catkin_make command not found. Make sure catkin is installed.
        exit /b 1
    )
    exit /b 0

:check_ros2
    :: Check for ros2 command
    where ros2 >nul 2>nul
    if %errorlevel% neq 0 (
        echo ros2 command not found. ROS 2 might not be installed or sourced.
        exit /b 1
    )
    :: Check for colcon command
    where colcon >nul 2>nul
    if %errorlevel% neq 0 (
        echo colcon command not found. Make sure colcon is installed.
        exit /b 1
    )
    exit /b 0

:check_boost
    :: Check if BOOST_ROOT is set
    if not defined BOOST_ROOT (
        echo BOOST_ROOT is not set. Boost might not be installed or configured properly.
        exit /b 1
    )
    :: Check for a common Boost header file
    if not exist "%BOOST_ROOT%\boost\version.hpp" (
        echo Boost headers not found. Please check your Boost installation.
        exit /b 1
    )
    echo Boost found at %BOOST_ROOT%
    exit /b 0
