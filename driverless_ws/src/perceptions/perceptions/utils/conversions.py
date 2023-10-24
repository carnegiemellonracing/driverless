
# ROS2 message types
from sensor_msgs.msg import Image, PointCloud2

# message to numpy conversion packages stolen from internet
from cv_bridge import CvBridge
import ros2_numpy as rnp

import numpy as np

def img_to_npy(img_msg: Image):
    bridge = CvBridge()
    return bridge.compressed_imgmsg_to_cv2(img_msg)

def pointcloud2_to_npy(pc2_msg: PointCloud2):
    points_raw = rnp.numpify(pc2_msg)

    points_arr = np.zeros((points_raw.shape[0], 6))

    points_arr[:,0] = points_raw['x'].reshape(-1)
    points_arr[:,1] = points_raw['y'].reshape(-1)
    points_arr[:,2] = points_raw['z'].reshape(-1)
    points_arr[:,3] = points_raw['intensity'].reshape(-1)
    points_arr[:,4] = points_raw['ring'].reshape(-1)
    points_arr[:,5] = points_raw['timestamp'].reshape(-1)

    return points_arr