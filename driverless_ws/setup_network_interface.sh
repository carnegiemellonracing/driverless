#!/bin/bash
set -e

################
# Constants
################
TRUE=0
FALSE=1
IPV4NM=24

################
# Functions
################

checkProgramExists() {
    if ! which "$1" >/dev/null; then
        echo "${1} NOT FOUND!" >&2
        REQ_CHECK=$FALSE
    fi
}

checkRequirements() {
    REQ_CHECK=$TRUE
    checkProgramExists ip
    checkProgramExists awk
    checkProgramExists sed
    checkProgramExists grep
    if [[ $(echo '123foobar#' | grep -oP 'foo\K\w+' || true) != bar ]]; then
        echo "grep was found, but it does not support Perl Compatible Regular Expressions (PCRE)." >&2
        REQ_CHECK=$FALSE
    fi
    checkProgramExists getopt
    getopt -T &>/dev/null
    if [[ $? -ne 4 ]]; then
        echo "This version of getopt is not supported." >&2
    fi
    return $REQ_CHECK
}

isFirewallEnabledUfw() {
    if which ufw &> /dev/null; then
        if ufw status | grep -Eq 'Status: inactive|Firewall not loaded'; then
            return $FALSE
        else
            return $TRUE
        fi
    fi
    return $FALSE
}

isFirewallEnabledIptables() {
    RET=$FALSE
    if which iptables &> /dev/null; then
        if iptables -S | grep -Eq -e '-P \w+ DROP'; then RET=$TRUE; fi
        if iptables -S | grep -E -e '-A (INPUT|FORWARD|OUTPUT).* -j (DROP|REJECT)' | grep -Eq -vi 'docker|virbr'; then RET=$TRUE; fi
    fi
    if which ip6tables &> /dev/null; then
        if ip6tables -S | grep -Eq -e '-P \w+ DROP'; then RET=$TRUE; fi
        if ip6tables -S | grep -E -e '-A (INPUT|FORWARD|OUTPUT).* -j (DROP|REJECT)' | grep -Eq -vi 'docker|virbr'; then RET=$TRUE; fi
    fi
    return $RET
}

getRealNetworkInterfaces() {
    while IFS= read -r -d '' iface; do
        if [[ -L /sys/class/net/${iface} ]]; then
            # filter out virtual interfaces
            if ! [[ "$(readlink "/sys/class/net/${iface}")" =~ /virtual/ ]]; then
                echo "$iface"
            fi
        else
            echo "$iface"
        fi
    done < <(find /sys/class/net -mindepth 1 -printf '%f\0')
}

promptConfirmation() {
    REPLY=
    while ! [[ $REPLY =~ [yYnN] ]]; do
        read -r -p "${1} [y]es/[n]o: "
    done
    if [[ $REPLY =~ [yY] ]]; then
        return $TRUE
    fi
    return $FALSE
}

getVirtualInterfaces() {
    ip -oneline link show | grep -oP "^\d+:\s+\K[^:@]+(?=@${1})" || true
}

getInterfaceNumber() {
    ip -oneline link show "$1" | grep -oP '^\d+'
}

getInterfaceState() {
    ip -oneline link show "$1" | grep -oP '\sstate\s\K[^\s]+'
}

waitInterfaceUp() {
    TIMEOUT=0
    while [[ $(getInterfaceState "$1") != UP ]]; do
        ((TIMEOUT++))
        if [[ ${TIMEOUT} -eq 5 ]]; then
            return $FALSE
        fi
        sleep 1
    done
    return $TRUE
}

checkPing() {
    TIMEOUT=0
    while ! ping -c 1 -s 24 -W 2 -w 5 "$1" >/dev/null; do
        ((TIMEOUT++))
        if [[ ${TIMEOUT} -eq 3 ]]; then
            return $FALSE
        fi
        sleep 1
    done
    return $TRUE
}

isManagedByNetworkManager() {
    if which nmcli &>/dev/null; then
        if [[ $(nmcli device status | grep "^${1}[[:space:]]" | awk '{print $3}') =~ unmanaged|unavailable ]]; then
            return $FALSE
        else
            return $TRUE
        fi
    else
        # network manager not installed
        return $FALSE
    fi
}

usage() {
    PROG=$(basename "$0")
    echo "Usage: ${PROG} -i IFACE [other options]...
Setup network interface for communication with an MOVIA B1 sensor.

Mandatory arguments to long options are mandatory for short options too.
 -i, --interface IFACE           Network interface connected to the sensor.
                                   Run ${PROG} -l for a list of possible interfaces.
 -s, --sensor-number NUM         Sensor number in range 1-199.
 -l, --list-network-interfaces   List possible network interfaces for --interface.
     --vlan-id  NUM              Number to iterate vlan interfaces with the prefix movia.
     --local-ip IPv4             Local IPv4 address of vlan network adapter.
     --remote-ip IPv4            Remote IPv4 address of the movia sensor.  
     --multicast-ip IPv6         Multicast IPv6 address of the movia data stream.
     --no-warn-firewall          Do not warn about a possibly enabled firewall.
     --no-ask-remove-interfaces  Silently remove virtual network interfaces.
     --no-ask-nm-unmanage        Silently set the interface to unmanaged via nmcli.
     --assume-interface-up       Skip check if the interface is up.
     --no-ping                   Skip sensor connection check.
     --force                     Caution! Skip all checks and confirmation prompts.
 -h, --help                      Print this help." >&2
}

################
# Parameters
################
WARN_FIREWALL=true # --no-warn-firewall
SILENT_REMOVE_VIF=false # --no-ask-remove-interfaces
SILENT_NM_UNMANAGE=false # --no-ask-nm-unmanage
CHECK_IF_UP=true # --assume-interface-up
CHECK_PING=true # --no-ping

IFACE=

SENSOR_NUMBER=1

VLAN_ID=101
LOCAL_IP="172.16.101.83"
REMOTE_IP="172.16.101.56"
MULTICAST_IPV6="ff02::1be0:1"

DIAG_VLAN_ID=2
DIAG_VLAN_IP="172.16.2.83"


################
# Early checks
################

# Check that this script is running as root user
if [[ $EUID -ne 0 ]]
  then echo "Please run this script as root" >&2
  exit 1
fi

# Check requirements to run this script are fulfilled
if ! checkRequirements; then
    echo "Please install the required tools to continue." >&2
    exit 1
fi

################
# Parse arguments
################

set +e
TMPARGS=$(getopt -o 'i:s:f:lh' \
    -l 'interface:,sensor-number:,fov:,list-network-interfaces,no-warn-firewall,no-ask-remove-interfaces,no-ask-nm-unmanage,assume-interface-up,no-ping,force,help' \
    -n "$0" -- "$@")
if [[ $? -ne 0 ]]; then
    usage
    exit 1
fi
set -e
eval set -- "$TMPARGS"
unset TMPARGS

while true; do
    case "$1" in
        -i|--interface) IFACE=$2; shift 2;;
        -s|--sensor-number) SENSOR_NUMBER=$2; shift 2;;
        -l|--list-network-interfaces) getRealNetworkInterfaces; exit 0;;
        --vlan-id) VLAN_ID=$2; shift 2;;
        --local-ip) LOCAL_IP=$2; shift 2;;
        --remote-ip) REMOTE_IP=$2; shift 2;;
        --multicast-ip) MULTICAST_IPV6=$2; shift 2;;
        --no-warn-firewall) WARN_FIREWALL=false; shift;;
        --no-ask-remove-interfaces) SILENT_REMOVE_VIF=true; shift;;
        --no-ask-nm-unmanage) SILENT_NM_UNMANAGE=true; shift;;
        --assume-interface-up) CHECK_IF_UP=false; shift;;
        --no-ping) CHECK_PING=false; shift;;
        --force)
            WARN_FIREWALL=false
            SILENT_REMOVE_VIF=true
            SILENT_NM_UNMANAGE=true
            CHECK_IF_UP=false
            CHECK_PING=false
            shift;;
        -h|--help) usage; exit 0;;
        --) shift; break;;
        *) usage; exit 1;;
    esac
done

if [[ -z $IFACE ]]; then
    echo "Error: Network interface not set." >&2
    usage
    exit 1
fi

if [[ $SENSOR_NUMBER =~ ^[0-9]+$] && [$SENSOR_NUMBER -ne 0 ]]; then
    REMOTE_IP_NUMBER=55 + $SENSOR_NUMBER 
    VLAN_ID=100 + $SENSOR_NUMBER
    LOCAL_IP="172.16.${VLAN_ID}.83"
    REMOTE_IP="172.16.${VLAN_ID}.${REMOTE_IP_NUMBER}"
    MULTICAST_IPV6="ff02::1be0:${SENSOR_NUMBER}"
fi

################
# Script "main"
################

# Check if there is a (typical) firewall active and warn about it
if $WARN_FIREWALL; then
    if isFirewallEnabledUfw || isFirewallEnabledIptables; then
        echo "The firewall on this system might be enabled!
Please disable it or ensure it does not block the following addresses/ports/protocols:
 - TCP/55000 to 172.16.0.0/16
 - UDP/12345 from ff02::1be0:0/120
 - IPv6 multicast
 - IPv4 ICMP
To silence this warning, add --no-warn-firewall to the command line."
        if ! promptConfirmation "Continue?"; then exit 1; fi
    fi
fi

# Temporarily disable network manager for the interface
if isManagedByNetworkManager "$IFACE"; then
    echo "The interface ${IFACE} is managed by NetworkManager, it could interfere with this script.
To always proceed without confirmation, add --no-ask-nm-unmanage to the command line."
    if $SILENT_NM_UNMANAGE || promptConfirmation "Disable managing of ${IFACE} temporarily?"; then
        nmcli device set "$IFACE" managed false
        echo "To enable NetworkManager again for ${IFACE}, run \"nmcli device set $IFACE managed true\" manually."
    else
        echo "Continuing with NetworkManager present. In case something does not work as expected, please disable NetworkManager for this interface."
    fi
fi

# Cleanup virtual interfaces that might already exist
VIRTUAL_IFACES="$(getVirtualInterfaces "$IFACE")"
if [[ -n $VIRTUAL_IFACES ]]; then
    ONLY_MOVIA_VIF=true
    for v in $VIRTUAL_IFACES; do
        if ! [[ $v =~ ^movia[0-9]+$ ]]; then ONLY_MOVIA_VIF=false; break; fi
    done
    echo "The interface ${IFACE} already has virtual interfaces ($(tr '\n' ' ' <<<"$VIRTUAL_IFACES")).
To proceed, the virtual interfaces have to be removed.
To always proceed without confirmation, add --no-ask-remove-interfaces to the command line."
    if $SILENT_REMOVE_VIF || $ONLY_MOVIA_VIF || promptConfirmation "Remove these interfaces?"; then
        for v in $VIRTUAL_IFACES; do
            ip link set dev "$v" down &>/dev/null
            ip link delete "$v" &>/dev/null
        done
    else
        exit 1
    fi
fi

# Cleanup interface
echo "Remove all addresses from ${IFACE}"
ip address flush dev "$IFACE"

# Disable link-local IPv6 address and set interface up
echo "Initialize interface ${IFACE}"
ip link set dev "$IFACE" addrgenmode none
ip link set dev "$IFACE" up

# Check the interface is up
if $CHECK_IF_UP; then
    echo "Waiting for interface ${IFACE} to be up..."
    if waitInterfaceUp "$IFACE"; then
        echo "Interface ${IFACE} is up"
    else
        echo "Interface ${IFACE} is not up, please connect all cables as written in the manual and supply power to all components."
        if promptConfirmation "Check again?"; then
            if waitInterfaceUp "$IFACE"; then
                echo "Interface ${IFACE} is up"
            else
                echo "Failed to bring interface ${IFACE} up, please re-check connections and power.
To disable this check and proceed without confirmation, add --assume-interface-up to the command line."
                exit 1
            fi
        else
            echo "Cancelled."
            exit 1
        fi
    fi
fi

# Create data VLAN interface
DATA_IFACE_NAME="mvis${VLAN_ID}"
echo "Create virtual interface for VLAN ${VLAN_ID}"
ip link add link "$IFACE" name "${DATA_IFACE_NAME}" type vlan id "${VLAN_ID}"
ip address add "${LOCAL_IP}/${IPV4NM}" brd + dev "$DATA_IFACE_NAME"
ip link set dev "$DATA_IFACE_NAME" up
DATA_IFACE_NUM="$(getInterfaceNumber "$DATA_IFACE_NAME")"

# Create diag VLAN interface
DIAG_IFACE_NAME="mvis${DIAG_VLAN_ID}"
echo "Create virtual interface for VLAN ${DIAG_VLAN_ID}"
ip link add link "$IFACE" name "$DIAG_IFACE_NAME" type vlan id "$DIAG_VLAN_ID"
ip address add "172.16.${DIAG_VLAN_ID}.83/${IPV4NM}" brd + dev "$DIAG_IFACE_NAME"
ip link set dev "$DIAG_IFACE_NAME" up

# Wait for data interface up and ping sensor
if $CHECK_IF_UP; then
    echo "Waiting for interface ${DATA_IFACE_NAME} to be up..."
    if waitInterfaceUp "$DATA_IFACE_NAME"; then
        echo "Interface ${DATA_IFACE_NAME} is up"
    else
        echo "Error: Interface ${DATA_IFACE_NAME} is not up.
To disable this check and proceed without confirmation, add --assume-interface-up to the command line." >&2
        exit 1
    fi
fi
if $CHECK_PING; then
    echo "Checking if sensor at ${REMOTE_IP} is alive..."
    if checkPing "$REMOTE_IP"; then
        echo "Sensor at ${REMOTE_IP} is alive."
    else
        echo "Error: Sensor at ${REMOTE_IP} did not respond!
To disable this check and proceed without confirmation, add --no-ping to the command line." >&2
        exit 1
    fi
fi


# Print results
echo "---------------------------------
Network interface setup is done.
The interface number of the sensor data interface is: ${DATA_IFACE_NUM}
Please use the following info to connect to the sensor:
  Sensor Type:                    MOVIA B1
  MOVIA B1 Connection
    Data IPv6 Connection
      Data Multicast IP Address:  ${MULTICAST_IPV6}%${DATA_IFACE_NUM}
      Local Port:                 12345
    SOME/IP Connection
      Remote IP Address:          ${REMOTE_IP}
      Remote Port:                55000
      Use Local Default Adapter:  no
        Local IP Address:         ${LOCAL_IP}
"
exit 0
