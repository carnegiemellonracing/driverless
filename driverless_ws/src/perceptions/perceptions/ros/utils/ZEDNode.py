''' ZEDNode.py

File contains implementation of publisher for stereocamera images and depth
maps for the purpose of perceptinos algorithms. Raw data is made accessible
via the ZEDSDK API which is then marshalled and published to the topics
listed out below
'''

import rclpy
from rclpy.node import Node
import torch
import numpy as np
import time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
 
from perceptions.topics import LEFT_IMAGE_TOPIC, RIGHT_IMAGE_TOPIC, XYZ_IMAGE_TOPIC, DEPTH_IMAGE_TOPIC, POINT_TOPIC
from perceptions.topics import LEFT2_IMAGE_TOPIC, RIGHT2_IMAGE_TOPIC, XYZ2_IMAGE_TOPIC, DEPTH2_IMAGE_TOPIC
from perceptions.zed import ZEDSDK

from eufs_msgs.msg import ConeArray #, DataFrame
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import cv2
from cv_bridge import CvBridge

BEST_EFFORT_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)

RELIABLE_QOS_PROFILE = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

CAMERA_PARAM = "camera"
ZED_STR = "zed"
ZED2_STR = "zed2"

# map cameras to their topics and serials numbers
CAMERA_INFO = {
    ZED_STR: (15080, LEFT_IMAGE_TOPIC, XYZ_IMAGE_TOPIC), 
    ZED2_STR: (27680008, LEFT2_IMAGE_TOPIC, XYZ2_IMAGE_TOPIC)
} # TODO: need serial numbers
        
class ZEDNode(Node):

    def __init__(self, camera_name=ZED2_STR):
        '''Initializes the ZED Node which publishes left color image and XYZ image

        Because there are multiple cameras for 22a, multiple ZEDNode classes 
        should be launched, one for each camera. Entry points for each camera are
        found at the bottom of this file and are registered in perceptions/setup.py

        Arguments:
            camera_name (str): name of camera to initialize (ZED_STR or ZED2_STR)
        '''

        super().__init__(f"{camera_name}_node")
        
        # ensure appropriate camera name
        self.camera_name = camera_name
        assert(self.camera_name in list(CAMERA_INFO.keys()))

        # unpack camera information
        self.serial_num, left_topic, xyz_topic = CAMERA_INFO[self.camera_name]
        
        # initialize all publishers
        # self.dataframe_publisher = self.create_publisher(msg_type=DataFrame,
        #                                                  topic='/DataFrame',
        #                                                  qos_profile=RELIABLE_QOS_PROFILE)
        self.left_publisher = self.create_publisher(msg_type=Image,
                                                     topic=left_topic,
                                                     qos_profile=RELIABLE_QOS_PROFILE)
        # self.right_publisher = self.create_publisher(msg_type=Image,
        #                                              topic=RIGHT_IMAGE_TOPIC,
        #                                              qos_profile=RELIABLE_QOS_PROFILE)
        # self.depth_publisher = self.create_publisher(msg_type=Image,
        #                                              topic=DEPTH_IMAGE_TOPIC,
        #                                              qos_profile=RELIABLE_QOS_PROFILE)
        self.xyz_publisher = self.create_publisher(msg_type=Image,
                                                   topic=xyz_topic,
                                                   qos_profile=RELIABLE_QOS_PROFILE)

        # initialize timer interval for publishing the data
        # TODO: frame rate higher than actual update rate
        frame_rate = 25
        self.data_syncer = self.create_timer(1/frame_rate, self.publish)

        # initialize the ZEDSDK API for receiving raw data
        self.zed = ZEDSDK(serial_num=self.serial_num)
        self.zed.open()

        self.bridge = CvBridge()
        self.frame_id = 0

    def publish(self):
        # try displaying the image

        s = time.time()

        # grab zed node data
        left, right, depth, xyz = self.zed.grab_data()

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = str(self.frame_id)
        self.frame_id += 1

        left_enc = self.bridge.cv2_to_imgmsg(left, encoding="passthrough", header=header)
        # right_enc = self.bridge.cv2_to_imgmsg(right, encoding="passthrough", header=header)
        # depth_enc = self.bridge.cv2_to_imgmsg(depth, encoding="passthrough", header=header)
        xyz_enc = self.bridge.cv2_to_imgmsg(xyz,encoding="32FC4", header=header)

        # publish the data
        self.left_publisher.publish(left_enc)
        # self.right_publisher.publish(right_enc)
        # self.depth_publisher.publis(depth_enc)
        self.xyz_publisher.publish(xyz_enc)

        t = time.time()
        print(f"Publishing data: {1000 * (t - s):.3f}ms (frame_id: {header.frame_id}, stamp: {header.stamp})")

def main_zed(args=None):
    # defaults to ZED2
    rclpy.init(args=args)
    minimal_subscriber = ZEDNode(camera_name=ZED_STR)

    rclpy.spin(minimal_subscriber)
    
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

def main_zed2(args=None):
    # defaults to ZED2
    rclpy.init(args=args)
    minimal_subscriber = ZEDNode(camera_name=ZED2_STR)

    rclpy.spin(minimal_subscriber)
    
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main_zed()