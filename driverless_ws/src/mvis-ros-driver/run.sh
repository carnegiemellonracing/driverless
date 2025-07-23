#!/bin/bash

# exit when any command fails
set -e

relativePathToScript=$_

# optional input given?
if [ $# -gt 0 ]; then
    POSITIONAL=()
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--hwid)
                HWID=$2
                shift # past key
                shift # past value
                ;;
            -p|--port)
                PORT=$2
                shift # past key
                shift # past value
                ;;
            -i|--ip)
                IP=$2
                shift # past key
                shift # past value
                ;;
            -l|--ldmi)
                LDMI=1
                shift # past key
                ;;
            *)
            POSITIONAL+=("$1") # Save unrecognized args in an array
            shift # past argument
            ;;
        esac
    done

    if [[ -z "$IP" && -z "$PORT" ]]; then
        echo "Usage: $0 --hwid <HWID> -i|--ip <IP> -p|--port <PORT> -l|--ldmi"
        echo "    HWID [--hwid|*]    Sensor HWID number or 'L'."
        echo "    IP   [-i|--ip|*]   Multicast IP address used for MOVIA L to override default.yaml."
        echo "    PORT [-p|--port|*] Port used for MOVIA L pointcloud to override default.yaml."
        echo "    [-l|--ldmi|*]      Set to receive ldmi raw data for MOVIA L pointcloud to override default.yaml."
        exit 0
    fi
fi

# ----------------------------

check_ros1_installation() {
    # Check if ROS_DISTRO is set
    if [ -z "$ROS_DISTRO" ]; then
        echo "ROS_DISTRO is not set. ROS 1 might not be installed or sourced."
        return 1
    fi

    # Check if ROS_VERSION is set and equals 1
    if [ -z "$ROS_VERSION" ] || [ "$ROS_VERSION" != "1" ]; then
        echo "ROS_VERSION is not set to 1. ROS 1 might not be installed or sourced."
        return 1
    fi

    # Check if essential ROS 1 commands are available
    if ! command -v roscore &> /dev/null; then
        echo "roscore command not found. ROS 1 might not be installed or sourced."
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
        echo "ROS_VERSION is not set to 2. ROS 2 might not be installed or sourced."
        return 1
    fi

    # Check if essential ROS 2 commands are available
    if ! command -v ros2 &> /dev/null; then
        echo "ros2 command not found. ROS 2 might not be installed or sourced."
        return 1
    fi

    echo "ROS 2 installation check passed."
    return 0
}

# ----------------------------

check_so_dependencies() {
    local so_file="$1"
    if [ ! -f "$so_file" ]; then
        echo "Error: $so_file does not exist."
        return 1
    fi

    # Run ldd and capture the output
    local ldd_output=$(ldd "$so_file")

    # Check if any dependencies are not found
    if echo "$ldd_output" | grep -q "not found"; then
        echo "Error: Some dependencies are missing for $so_file"
        echo "Missing dependencies:"
        echo "$ldd_output" | grep "not found"
        return 1
    else
        echo "All dependencies are found for $so_file"
        return 0
    fi
}

# ----------------------------

# Function to ask for sensor type and HWID number if applicable
configure_sensor() {
    PS3="Please select the sensor type: "
    options=("MOVIA L" "HWID")
    select opt in "${options[@]}"
    do
        case $opt in
            "MOVIA L")
                sensor_type="L"
                break
                ;;
            "HWID")
                while true; do
                    read -p "Please enter the HWID number: " hwid_number
                    if [[ $hwid_number =~ ^[0-9]+$ ]]; then
                        sensor_type="$hwid_number"
                        break
                    else
                        echo "Invalid input. Please enter a valid number." > /dev/tty
                    fi
                done
                break
                ;;
            *)
                echo "Invalid option. Please try again." > /dev/tty
                ;;
        esac
    done

    # Ask for multicast IP address
    default_ip="224.100.100.20"
    while true; do
        read -p "Multicast IP (default: ${default_ip}): " multicast_ip
        multicast_ip=${multicast_ip:-${default_ip}}  # Set default if empty
        # IPv4 regex
        ipv4_regex='^([0-9]{1,3}\.){3}[0-9]{1,3}$'
        # IPv6 regex (simplified, allows most common formats)
        ipv6_regex='^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$|^::([0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}$|^[0-9a-fA-F]{1,4}::([0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4}$|^[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}::([0-9a-fA-F]{1,4}:){0,4}[0-9a-fA-F]{1,4}$|^([0-9a-fA-F]{1,4}:){2}:([0-9a-fA-F]{1,4}:){0,3}[0-9a-fA-F]{1,4}$|^([0-9a-fA-F]{1,4}:){3}:([0-9a-fA-F]{1,4}:){0,2}[0-9a-fA-F]{1,4}$|^([0-9a-fA-F]{1,4}:){4}:([0-9a-fA-F]{1,4}:)?[0-9a-fA-F]{1,4}$|^([0-9a-fA-F]{1,4}:){5}:[0-9a-fA-F]{1,4}$|^([0-9a-fA-F]{1,4}:){6}:[0-9a-fA-F]{1,4}$'

        if [[ $multicast_ip =~ $ipv4_regex ]]; then
            IFS='.' read -r -a ip_parts <<< "$multicast_ip"
            valid_ip=true
            for part in "${ip_parts[@]}"; do
                if (( part < 0 || part > 255 )); then
                    valid_ip=false
                    break
                fi
            done

            if [[ ${ip_parts[0]} -ge 224 && ${ip_parts[0]} -le 239 ]]; then
                # This is a multicast address, so we need to check the route
                echo "Multicast address detected. Checking multicast route..."
                if ! ip route show | grep -q "224.0.0.0/4"; then
                    echo "Warning: No multicast route found." > /dev/tty
                    echo "You may need to add a route using:" > /dev/tty
                    echo "sudo ip route add 224.0.0.0/4 dev <your_interface>" > /dev/tty
                    echo "Replace <your_interface> with your network interface name." > /dev/tty
                    echo "Continue anyway? (y/n)" > /dev/tty
                    read -r continue_anyway
                    if [[ $continue_anyway != "y" ]]; then
                        valid_ip=false
                    fi
                else
                    echo "Multicast route found." > /dev/tty
                fi
            fi

            if $valid_ip; then
                break
            fi
        elif [[ $multicast_ip =~ $ipv6_regex ]]; then
            break
        else
            echo "Invalid IP address. Please enter a valid multicast IPv4 or IPv6 address." > /dev/tty
        fi
    done

    # Ask for port number
    default_port=30000
    while true; do
        read -p "Enter the port number (default: ${default_port}): " port_number
        port_number=${port_number:-${default_port}}  # Set default if empty
        if [[ $port_number =~ ^[0-9]+$ ]] && (( port_number >= 1 && port_number <= 65535 )); then
            break
        else
            echo "Invalid port number. Please enter a number between 1 and 65535." > /dev/tty
        fi
    done

    echo "$sensor_type"
    echo "$multicast_ip"
    echo "$port_number"
}

# ----------------------------

relativePathToScript=$(dirname "$0")

if check_ros1_installation; then
    echo "ROS 1 ${ROS_DISTRO} is properly installed and sourced. Proceeding with ROS1 launch..."

    cd "$relativePathToScript/ros" || exit 3

    source ./install/setup.sh

    export LD_LIBRARY_PATH="$PWD/install/lib/movia:$LD_LIBRARY_PATH"

    check_so_dependencies "./install/lib/movia/movia"
    check_so_dependencies "./install/lib/movia/libmovia-device-plugin.so"
    check_so_dependencies "./install/lib/movia/libmovia-interpreter-plugin.so"
    check_so_dependencies "./install/lib/movia/libthirdparty-pcap-plugin.so"

    # construct the run command
    runCommand="roslaunch movia movia.launch"

    if [ -n "$HWID" ]; then
        runCommand+=" hwid:=\"'$HWID'\""
   else
      # if no movia L or HWID - ask for sensor type
      sensor_info=$(configure_sensor | tail -n 3)
      HWID=$(echo "$sensor_info" | head -n 1)
      IP=$(echo "$sensor_info" | tail -n 2 | head -n 1)
      PORT=$(echo "$sensor_info" | tail -n 1)

      runCommand+=" hwid:=\"'$HWID'\""
    fi

    # add IP and PORT parameters only if they were provided
    if [ -n "$IP" ]; then
        runCommand+=" multicast_ip:=\"'$IP'\""
    fi

    if [ -n "$PORT" ]; then
        runCommand+=" port:=$PORT"
    fi

    if [ -n "$LDMI" ]; then
        runCommand+=" ldmi_raw:=true"
    fi

    # execute the command
    eval $runCommand

elif check_ros2_installation; then
    echo "ROS 2 ${ROS_DISTRO} is properly installed and sourced. Proceeding with ROS2 run ..."

    cd "$relativePathToScript/ros2" || exit 3

    source ./install/setup.sh

    export LD_LIBRARY_PATH="./install/movia/lib/movia:$LD_LIBRARY_PATH"

    check_so_dependencies "./install/movia/lib/movia/movia"
    check_so_dependencies "./install/movia/lib/movia/libmovia-device-plugin.so"
    check_so_dependencies "./install/movia/lib/movia/libmovia-interpreter-plugin.so"
    check_so_dependencies "./install/movia/lib/movia/libthirdparty-pcap-plugin.so"

    # construct the run command
    runCommand="ros2 run movia movia --ros-args --params-file ./config/default.yaml"
#debug    runCommand="ros2 run --prefix 'gdbserver localhost:3000' movia movia --ros-args --params-file ./config/default.yaml"

    if [ -n "$HWID" ]; then
        runCommand+=" --param hwid:=\"'$HWID'\""
   else
      # if no movia L or HWID - ask for sensor type
      sensor_info=$(configure_sensor | tail -n 3)
      HWID=$(echo "$sensor_info" | head -n 1)
      IP=$(echo "$sensor_info" | tail -n 2 | head -n 1)
      PORT=$(echo "$sensor_info" | tail -n 1)

      runCommand+=" --param  hwid:=\"'$HWID'\""
    fi

    # add IP and PORT parameters only if they were provided
    if [ -n "$IP" ]; then
        runCommand+=" --param multicast_ip:=\"'$IP'\""
    fi

    if [ -n "$PORT" ]; then
        runCommand+=" --param port:=$PORT"
    fi

    if [ -n "$LDMI" ]; then
        runCommand+=" --param ldmi_raw:=true"
    fi

    # execute the command
    eval $runCommand
else
    echo "ROS 1 and 2 installation check failed. Please install or source ROS before building."
    exit 1
fi
