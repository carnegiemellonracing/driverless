from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

# CollectionNode Topics
ZED_LEFT = '/zed2/zed_node/rgb_raw/image_raw_color'
VLP16 = '/velodyne_points'
ZED_POINTS = '/zed2/zed_node/point_cloud/cloud_registered'
SBG = '/sbg/gps_pos'
IMU = '/sbg/imu_data'
DATA_TIME_SYNC = '/data_time_sync'

# SimCollectionNode Topics
SIM_ZED_LEFT = '/zed/left/image_rect_color'
SIM_VLP16 = '/velodyne_points'
SIM_ZED_POINTS = '/zed/points'
SIM_GT_CONES = '/ground_truth/cones'
SIM_IMU = '/imu'
SIM_FUSION_CONES = '/fusion/cones/viz'
SIM_ZED_DEPTH = '/zed/depth/image_raw'
SIM_DATA_TIME_SYNC = '/sim_data_time_sync'

# QOS Profiles
BEST_EFFORT_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)

RELIABLE_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.RELIABLE,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)

QUEUE_SIZE = 30
APPROX_DELAY = 0.5 # sec
