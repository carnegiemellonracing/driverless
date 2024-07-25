# topic names that we are reading from for stereocamera data defined by us
LEFT_IMAGE_TOPIC = "/zedsdk_left_color_image"
RIGHT_IMAGE_TOPIC = "/zedsdk_right_color_image"
XYZ_IMAGE_TOPIC = "/zedsdk_point_cloud_image"
DEPTH_IMAGE_TOPIC = "/zedsdk_depth_image"

# topic names that we are reading from for stereocamera data defined by us for ZED 2
LEFT2_IMAGE_TOPIC = "/zedsdk2_left_color_image"
RIGHT2_IMAGE_TOPIC = "/zedsdk2_right_color_image"
XYZ2_IMAGE_TOPIC = "/zedsdk2_point_cloud_image"
DEPTH2_IMAGE_TOPIC = "/zedsdk2_depth_image"

# topic names for lidar data reading from Hesai lidar
POINT_TOPIC = "/lidar_points"

# topic names for publishing cones
LIDAR_CONE_TOPIC = "/lidar_node_cones"
YOLOV5_ZED_CONE_TOPIC = "/yolov5_zed_node_cones"
YOLOV5_ZED2_CONE_TOPIC = "/yolov5_zed2_node_cones"
PERC_CONE_TOPIC = "/perc_cones"

# topic names for motion modeling
TWIST_TOPIC = "/filter/twist"
QUAT_TOPIC = "/filter/quaternion"

#Camera info
CAMERA_PARAM = "camera"
ZED_STR = "zed"
ZED2_STR = "zed2"

# map cameras to their topics and serials numbers
CAMERA_INFO = {
    ZED_STR: (15080, LEFT_IMAGE_TOPIC, XYZ_IMAGE_TOPIC), 
    ZED2_STR: (27680008, LEFT2_IMAGE_TOPIC, XYZ2_IMAGE_TOPIC)
} # TODO: need serial numbers

# topic name for synced camera/lidar and odometry data
SYNCED_DATA_TOPIC = "/synced_data"
