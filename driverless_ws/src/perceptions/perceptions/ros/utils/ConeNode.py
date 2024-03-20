# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

# cone datatype for ROS and perc22a
from sensor_msgs.msg import PointCloud2
from eufs_msgs.msg import ConeArray
from perc22a.predictors.utils.cones import Cones
from perc22a.predictors.utils.transform.transform import PoseTransformations
import perceptions.ros.utils.conversions as conv

from perc22a.utils.Timer import Timer

# perceptions Library visualization functions (for 3D data)
# from perc22a.predictors.utils.vis.Vis3D import Vis3D
from perc22a.predictors.utils.vis.Vis2D import Vis2D

from perceptions.topics import \
    YOLOV5_ZED_CONE_TOPIC, \
    YOLOV5_ZED2_CONE_TOPIC, \
    LIDAR_CONE_TOPIC, \
    PERC_CONE_TOPIC, \
    POINT_TOPIC

# general imports
import cv2
import numpy as np

# configure QOS profile
BEST_EFFORT_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)
RELIABLE_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.RELIABLE,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)

CONE_NODE_NAME = "cone_node"
PUBLISH_FPS = 10
VIS_UPDATE_FPS = 25
MAX_ZED_CONE_RANGE = 12.5

def within_range(coords):
    return np.linalg.norm(np.array(coords)) <= MAX_ZED_CONE_RANGE

def zero_cone_z(coords):
    return [coords[0], coords[1], 0]

class ConeNode(Node):

    def __init__(self, debug=True, visualize_points=True):
        super().__init__(CONE_NODE_NAME)

        self.cones = Cones()

        # initialize all cone subscribers
        self.create_subscription(ConeArray, YOLOV5_ZED_CONE_TOPIC, self.yolov5_zed_cone_callback, qos_profile=RELIABLE_QOS_PROFILE)
        self.create_subscription(ConeArray, YOLOV5_ZED2_CONE_TOPIC, self.yolov5_zed2_cone_callback, qos_profile=RELIABLE_QOS_PROFILE)
        self.create_subscription(ConeArray, LIDAR_CONE_TOPIC, self.lidar_cone_callback, qos_profile=RELIABLE_QOS_PROFILE)

        # initialize point cloud subscriber for visualization (and pose transformer)
        if debug and visualize_points:
            # self.create_subscription(PointCloud2, POINT_TOPIC, self.point_cloud_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
            self.pose_transformer = PoseTransformations()

        # initialize cone publisher
        self.publish_timer = self.create_timer(1/PUBLISH_FPS, self.publish_cones)
        self.cone_publisher = self.create_publisher(ConeArray, PERC_CONE_TOPIC, qos_profile=RELIABLE_QOS_PROFILE)

        # deubgging mode visualizer
        if debug:
#            self.vis3D = Vis3D()
            self.vis2D = Vis2D()
            self.display_timer = self.create_timer(1/VIS_UPDATE_FPS, self.update_vis)

        # if debugging, initialize visualizer
        self.debug = debug
        self.visualize_points = visualize_points
        self.stop = False

        # sufficient cone policy implementation
        self.got_zed_left = False
        self.got_zed_right = False

        return

    def update_vis(self):
        # update and interact with vis
 #       self.vis3D.update()
        self.vis2D.update()

        return
    
    def point_cloud_callback(self, msg):
        points = conv.pointcloud2_to_npy(msg)[:, :3]

        points = points[np.any(points != 0, axis=1)]
        
        points = points[:, :3]
        points = points[:, [1, 0, 2]]
        points[:, 0] = -points[:, 0]
        
        t = Timer()

        t.start("pose")
        points = self.pose_transformer.to_origin("lidar", points, inverse=False)
        t.end("pose")

  #      self.vis3D.set_points(points)
        self.vis2D.set_points(points)


    def yolov5_zed_cone_callback(self, msg):
        '''receive cones from yolov5_zed_node predictor'''
        cones = conv.msg_to_cones(msg)
        cones.filter(within_range)
        self.cones.add_cones(cones)

        self.got_zed_right = True

        return
    
    def yolov5_zed2_cone_callback(self, msg):
        '''receive cones from yolov5_zed2_node predictor'''
        cones = conv.msg_to_cones(msg)
        cones.filter(within_range)
        self.cones.add_cones(cones)

        self.got_zed_left = True

        return

    def lidar_cone_callback(self, msg):
        '''receive cones from lidar_node predictor'''
        cones = conv.msg_to_cones(msg)
        self.cones.add_cones(cones)

        return
    
    def sufficient_cones(self):
        return len(self.cones) > 0 and self.got_zed_left and self.got_zed_right

    def flush_cones(self):
        self.cones = Cones()
        self.got_zed_left = False
        self.got_zed_right = False
        return

    def publish_cones(self):

        if not self.sufficient_cones():
            self.get_logger().warn(f"Not got sufficient cones")
            return 

        # update visualizer
        if self.debug:
   #         self.vis3D.set_cones(self.cones)
            self.vis2D.set_cones(self.cones)

        # reset the height of the cone to 0
        self.cones.map(zero_cone_z)

        # publish cones
        print(f"publishing {len(self.cones)} cones")
        msg = conv.cones_to_msg(self.cones)
        self.cone_publisher.publish(msg)
            
        # flush cones
        self.flush_cones()
        
        return
    
def start_cone_node(args=None, debug=False):
    rclpy.init(args=args)

    cone_node = ConeNode(debug=debug)

    rclpy.spin(cone_node)

    cone_node.destroy_node()
    rclpy.shutdown()

    return

def main(args=None):
    start_cone_node(args=args, debug=False)
    return

def main_debug(args=None):
    start_cone_node(args=args, debug=True)
    return

if __name__ == "__main__":
    main()
