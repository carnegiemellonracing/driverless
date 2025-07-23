@echo off
setlocal enabledelayedexpansion

goto :main

:configure_sensor
:: Select sensor type
echo Please select the sensor type:
echo 1) MOVIA L
echo 2) HWID
set /p sensor_choice="Enter your choice (1 or 2): "

if "%sensor_choice%"=="1" (
    set "HWID=L"
) else if "%sensor_choice%"=="2" (
    :hwid_input
    set /p HWID="Please enter the HWID number: "
    echo !HWID!| findstr /r "^[0-9]*$" >nul
    if errorlevel 1 (
        echo Invalid input. Please enter a valid number.
        goto :hwid_input
    )
) else (
    echo Invalid option. Please try again.
    goto :configure_sensor
)

:: Ask for multicast IP address
set "default_ip=224.100.100.20"
:ip_input
set /p "IP=Multicast IP (default: %default_ip%): "
if not defined IP set "IP=%default_ip%"

:: Simple IPv4 validation (not as comprehensive as the bash version)
echo !IP!| findstr /r "^[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*$" >nul
if errorlevel 1 (
    echo Invalid IP address. Please enter a valid IPv4 address.
    goto :ip_input
)

:: Check if it's a multicast address
for /f "tokens=1 delims=." %%a in ("!IP!") do set first_octet=%%a
if !first_octet! geq 224 if !first_octet! leq 239 (
    echo Multicast address detected. Checking multicast route...
    route print | findstr /C:"224.0.0.0" >nul
    if errorlevel 1 (
        echo Warning: No multicast route found.
        echo You may need to add a route using:
        echo route add 224.0.0.0 MASK 255.0.0.0 ^<your_interface_ip^>
        echo Replace ^<your_interface_ip^> with your network interface IP.
        set /p continue_anyway="Continue anyway? (y/n): "
        if /i not "!continue_anyway!"=="y" goto :ip_input
    ) else (
        echo Multicast route found.
    )
)

:: Ask for port number
set "default_port=30000"
:port_input
set /p "PORT=Enter the port number (default: %default_port%): "
if not defined PORT set "PORT=%default_port%"
echo !PORT!| findstr /r "^[0-9]*$" >nul
if errorlevel 1 (
    echo Invalid port number. Please enter a number between 1 and 65535.
    goto :port_input
)
if !PORT! lss 1 (
    echo Invalid port number. Please enter a number between 1 and 65535.
    goto :port_input
)
if !PORT! gtr 65535 (
    echo Invalid port number. Please enter a number between 1 and 65535.
    goto :port_input
)

exit /b

:main
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

:: Parse command-line arguments
:parse_args
if "%~1"=="" goto end_parse_args
if /i "%~1"=="-d" (
    set "HWID=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--hwid" (
    set "HWID=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="-p" (
    set "PORT=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--port" (
    set "PORT=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="-i" (
    set "IP=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--ip" (
    set "IP=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="-l" (
    set "LDMI=1"
    shift
    goto parse_args
)
if /i "%~1"=="--ldmi" (
    set "LDMI=1"
    shift
    goto parse_args
)
if /i "%~1"=="-h" (
    goto :show_help
)
if /i "%~1"=="--help" (
    goto :show_help
)
shift
goto parse_args
:end_parse_args

if not defined HWID (
    :: If no arguments provided, use wizard
    call :configure_sensor
)

:: Prepare arguments for ROS commands
set "ROS_ARGS="

:: Check ROS version
if "%ROS_VERSION%"=="1" (
    call :check_ros1
    
    echo ROS 1 %ROS_DISTRO% is properly installed and sourced. Proceeding with ros launch ...
    
    cd ./ros/

    call .\install\setup.bat

    if defined HWID set "ROS_ARGS=!ROS_ARGS! hwid:=""!HWID!"""
    if defined IP set "ROS_ARGS=!ROS_ARGS! multicast_ip:=!IP!"
    if defined PORT set "ROS_ARGS=!ROS_ARGS! port:=!PORT!"
    if defined LDMI set "ROS_ARGS=!ROS_ARGS! ldmi_raw:=True"

    roslaunch movia movia.launch !ROS_ARGS!
    
) else if "%ROS_VERSION%"=="2" (
    call :check_ros2

    echo ROS 2 %ROS_DISTRO% is properly installed and sourced. Proceeding with ros run ...

    cd ./ros2/

    call .\install\setup.bat

    if defined HWID set "ROS_ARGS=!ROS_ARGS! --param hwid:=""!HWID!"""
    if defined IP set "ROS_ARGS=!ROS_ARGS! --param multicast_ip:=!IP!"
    if defined PORT set "ROS_ARGS=!ROS_ARGS! --param port:=!PORT!"
    if defined LDMI set "ROS_ARGS=!ROS_ARGS! --param ldmi_raw:=True"

    ros2 run movia movia --ros-args --params-file ./config/default.yaml !ROS_ARGS!
    
) else (
    echo Unknown ROS_VERSION: %ROS_VERSION%
    exit /b 1
)

exit /b 0

:show_help
echo Usage: run.bat [OPTIONS]
echo Options:
echo   -d, --hwid ^<HWID^>      Sensor HWID number or 'L' for MOVIA L.
echo   -i, --ip ^<IP^>          Multicast IP address (overrides default.yaml).
echo   -p, --port ^<PORT^>      Port for MOVIA L pointcloud (overrides default.yaml).
echo   -l, --ldmi               Enable LDMI raw data reception (overrides default.yaml).
echo   -h, --help               Display this help information.
exit /b 0

:check_ros1
    where roscore >nul 2>nul
    if %errorlevel% neq 0 (
        echo roscore command not found. ROS 1 might not be installed or sourced.
        exit /b 1
    )
    exit /b 0

:check_ros2
    where ros2 >nul 2>nul
    if %errorlevel% neq 0 (
        echo ros2 command not found. ROS 2 might not be installed or sourced.
        exit /b 1
    )
    exit /b 0
