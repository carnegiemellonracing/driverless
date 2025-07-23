# mvis-ros-driver

This is the MVIS ROS and ROS2 node to integrate a MOVIA Solid State Flash LiDAR sensor into an existing ROS or ROS2 environment.

Requirements:
-------------
- ROS1 or ROS 2 installed and sourced. Or use a ROS docker.

Usage
-----
```
./build.sh
```

Continue with the next step after a successful build. If the build script fails please check the error messages.

```
./run.sh [OPTIONS]
```

Only after success start the run script:
If no options are provided, the script will start an interactive configuration "wizard" that:
- Prompts the user to select the sensor type
- Asks for the multicast IP address of the sensor (with a default option)
- Requests the port number (with a default option)

It might be required to set up a route for multicast packages for your network interface:
```
sudo ip route add 224.0.0.0/4 dev <your_interface>
```

Options:
--------
```
  -d, --hwid <HWID>   Specify the sensor's HWID (L for MOVIA L).

  -i, --ip <IP>       Specify the sensor's IP address.
                      Default: 224.100.100.20

  -p, --port <PORT>   Specify the sensor's port.
                      Default: 30000

  -l, --ldmi          Enable LDMI raw data processing for MOVIA L pointcloud.

  -h, --help          Display this help information.
```

Please note that you can also configure (these and more) settings in the yaml configuration file or with rosparam set!

Examples:
---------
```
./run.sh -d L -i 224.100.100.20 -p 30000
``` 
```
./run.sh -d 19 -i 224.100.100.21 -p 31000 -l
```