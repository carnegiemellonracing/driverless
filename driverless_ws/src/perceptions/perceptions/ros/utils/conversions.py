
# ROS2 message types
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2
from interfaces.msg import ConeArray
from geometry_msgs.msg import Point
from geometry_msgs.msg import QuaternionStamped

# perc22a Cone class
from perc22a.predictors.utils.cones import Cones

# message to numpy conversion packages stolen from internet
from cv_bridge import CvBridge
import ros2_numpy as rnp

import numpy as np

# helper functions for performing conversions
def _cone_to_msg_arr(cone_arr):
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

def _msg_to_cone_arr(msg_arr):
    cone_arr = np.zeros((len(msg_arr), 3))
    
    for i, point_msg in enumerate(msg_arr):
        cone_arr[i, 0] = point_msg.x
        cone_arr[i, 1] = point_msg.y
        cone_arr[i, 2] = point_msg.z

    return cone_arr

# primary interface for conversions.py
def img_to_npy(img_msg: Image):
    bridge = CvBridge()
    return bridge.imgmsg_to_cv2(img_msg)

def pointcloud2_to_npy(pc2_msg: PointCloud2):
    points_raw = rnp.numpify(pc2_msg)

    points_arr = np.zeros((points_raw.shape[0], 3))

    points_arr[:,0] = points_raw['x'].reshape(-1)
    points_arr[:,1] = points_raw['y'].reshape(-1)
    points_arr[:,2] = points_raw['z'].reshape(-1)
    # points_arr[:,3] = points_raw['intensity'].reshape(-1)
    # points_arr[:,4] = points_raw['ring'].reshape(-1)
    # points_arr[:,5] = points_raw['timestamp'].reshape(-1)

    return points_arr

def npy_to_pointcloud2(pc):
    pc_array = np.zeros(len(pc), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32),
    ])
    pc_array['x'] = pc[:, 0]
    pc_array['y'] = pc[:, 1]
    pc_array['z'] = pc[:, 2]
    pc_array['intensity'] = 255

    pc_msg = rnp.msgify(PointCloud2, pc_array)
    return pc_msg

def cones_to_msg(cones: Cones) -> ConeArray:
    '''convert perc22a Cones datatype to ConeArray ROS2 msg type'''
    
    cones_msg = ConeArray()
    blue_cones, yellow_cones, orange_cones = cones.to_numpy()

    cones_msg.blue_cones = _cone_to_msg_arr(blue_cones)
    cones_msg.yellow_cones = _cone_to_msg_arr(yellow_cones)
    cones_msg.orange_cones = _cone_to_msg_arr(orange_cones)
    
    return cones_msg

def msg_to_cones(msg: ConeArray) -> Cones:

    return Cones.from_numpy(
        _msg_to_cone_arr(msg.blue_cones),
        _msg_to_cone_arr(msg.yellow_cones),
        _msg_to_cone_arr(msg.orange_cones)
    )

def ms_since_time(now: Time, stamp: Time):
    delta_nanos = now.nanoseconds - stamp.nanoseconds
    delta_ms = int(delta_nanos / 1e6)
    return delta_ms