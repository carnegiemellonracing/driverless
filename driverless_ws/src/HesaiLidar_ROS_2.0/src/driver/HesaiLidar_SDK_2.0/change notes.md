# HesaiLidar_SDK_2.0

################################################################################
Friday, June 9th, 2023 16:45 
## version
V2.0.1

## modify
1. first update
2. fix AT128 frame segmentation bug

################################################################################
Monday, October 16th, 2023 11:00 
## version
V2.0.4

## modify
1. support ET25 

################################################################################
Wednesday, October 25th, 2023 20:00 
## version
V2.0.5

## modify
1. Fix bug in decode OT128 timestamp

################################################################################
Monday, January 15th, 2024 21:30
## version
V2.0.6

## modify
1. Add point cloud timestamp usage type configuration
2. Add ptc connection configuration parameters
3. Add et angle global correction
4. Fix the problem that coordinate transformation parameters do not take effect.
5. Added unified memory management in the LidarDecodedFrame class
6. Support P128 and XT32 to set the speed and standby through the configuration file
7. Support the configuration of common lidar configuration items
8. Add support for AT128 to intercept FOV display


################################################################################
Saturday April 13th, 2024 20:10:00 CST
## version
V2.0.7

## modify
1. Fix gpu compile problem
2. Resolve compile warnings, inluding cpu compile and gpu compile
3. ROS add topics, support angle correction data, ptp data, loss bag count data
4. Support AT128P inversion
5. Update XT16/XT32/XT32M2X/OT128 firetime correction file
6. Update AT128 and AT128P angle correction file
7. Resolve windows compile error


################################################################################
Wednesday September 25th, 2024 11:04:33 CST
## version
V2.0.8

## modify
1. support ET25, ET30, ET30H, ATX parsing.
2. fix the problem that the length of point cloud and the parsing result are not checked, which leads to the problem that the number of packets and points in individual frames is too high.
3. support filtering point cloud and fault message data from specified source ports when parsing multiple lidar.
4. Modify the optical centre correction scheme of mechanical lidar, you can configure the corresponding parameters to enable the optical centre correction of different types of lidar.
5. fix the problem that the logger class starts repeatedly and causes a crash.
6. fix the problem that the shared data is not protected when multi-threaded parsing.
7. support XT16 and XT32 to correct the point cloud S stratification.
8. improve the GPU function of all models of lidar and optimise the copying logic.
9. support the parsing of AT128/AT128P and ATX's fault message.
10. support packet loss detection and point cloud timestamp fallback detection.
11. adjust the startup sequence, do not need to wait for the PTC to connect successfully before starting to parse the fault message.
12. remove LidarDecodedPacket type, use LidarDecodedFrame type to store intermediate results, and optimise the parsing process.
13. fix the bug of pcap reading overflow.
14. fix the problem that mechanical lidar can't be divided into frames after FOV is turned on.
15. use a separate thread for PTC initialisation, and lock when PTC is called to prevent data from being messed up.
16. Refuse to parse the point cloud when the angle calibration file is not loaded to prevent parsing data anomaly.
17. Use log module to record logs.
18. update Firetime files of OT128, Pandar40P, Pandar64, support Firetime files of ATX, PandarQT, and angle calibration files of ATX 4.2.
19. Fix the bug that the change of LidarPoint type to no timestamp leads to the exception of obtaining timestamp directly.
20. Compatible with OT firetimes analysis
21. Support reading ATX .dat format firetimes.
22. Fix the problem of not considering forward and reverse rotation when fixing mechanical lidar firetimes.
