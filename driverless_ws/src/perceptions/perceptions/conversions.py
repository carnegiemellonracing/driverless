
# ROS2 message types
from sensor_msgs.msg import Image, PointCloud2
from eufs_msgs.msg import ConeArray
from geometry_msgs.msg import Point

# perc22a Cone class
from perc22a.predictors.utils.cones import Cones

# message to numpy conversion packages stolen from internet
from cv_bridge import CvBridge
import ros2_numpy as rnp

import numpy as np

# helper functions for performing conversions
def _cone_msg_arr(cone_arr):
    '''convert (N, 3) array to ConeWithCovariance[]'''
    arr_msg = []
    for i in range(cone_arr.shape[0]):
        x, y, z = cone_arr[i, :]
 
        msg = Point()
        msg.x = float(x)
        msg.y = float(y)
        msg.z = float(z)

        arr_msg.append(msg)
        
    return arr_msg

# primary interface for conversions.py
def img_to_npy(img_msg: Image):
    bridge = CvBridge()
    return bridge.imgmsg_to_cv2(img_msg)

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

def cones_to_msg(cones: Cones) -> ConeArray:
    '''convert perc22a Cones datatype to ConeArray ROS2 msg type'''
    
    cones_msg = ConeArray()
    blue_cones, yellow_cones, orange_cones = cones.get_cones()

    cones_msg.blue_cones = _cone_msg_arr(blue_cones)
    cones_msg.yellow_cones = _cone_msg_arr(yellow_cones)
    cones_msg.orange_cones = _cone_msg_arr(orange_cones)
    
    return cones_msg