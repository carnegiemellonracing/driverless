# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

# cone datatype for ROS and perc22a
from eufs_msgs.msg import ConeArray
from perc22a.predictors.utils.cones import Cones
import perceptions.ros.utils.conversions as conv

# perceptions Library visualization functions (for 3D data)
from perc22a.predictors.utils.vis.Vis3D import Vis3D

from perceptions.topics import \
    YOLOV5_ZED_CONE_TOPIC, \
    YOLOV5_ZED2_CONE_TOPIC, \
    LIDAR_CONE_TOPIC, \
    PERC_CONE_TOPIC

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

class ConeNode(Node):

    def __init__(self, debug=False):
        super().__init__(CONE_NODE_NAME)

        self.cones = None

        # initialize all cone subscribers
        # self.create_subscription(ConeArray, YOLOV5_ZED_CONE_TOPIC, None, qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.create_subscription(ConeArray, LIDAR_CONE_TOPIC, self.lidar_cone_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)

        # initialize cone publisher
        self.publish_timer = self.create_timer(1/PUBLISH_FPS, self.publish_cones)

        # deubgging mode visualizer
        if debug:
            self.vis = Vis3D()
            self.display_timer = self.create_timer(1/VIS_UPDATE_FPS, self.update_vis)


        # if debugging, initialize visualizer
        self.debug = debug

    def update_vis(self):
        # update and interact with vis
        self.vis.update()

    def yolov5_zed_cone_callback(self, msg):

        print("Got cone")

    def lidar_cone_callback(self, msg):
        cones = conv.msg_to_cones(msg)
        self.cones = cones

    def publish_cones(self):

        if self.debug:
            self.vis.set_cones(self.cones)
        pass

def main(args=None):
    rclpy.init(args=args)

    cone_node = ConeNode(debug=True)

    rclpy.spin(cone_node)

    cone_node.destroy_node()
    rclpy.shutdown()

    return

if __name__ == "__main__":
    main()
