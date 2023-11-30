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


from perceptions.ros.utils.topics import LEFT_IMAGE_TOPIC, RIGHT_IMAGE_TOPIC, XYZ_IMAGE_TOPIC, DEPTH_IMAGE_TOPIC, POINT_TOPIC
from perceptions.ros.utils.zed import ZEDSDK

from eufs_msgs.msg import ConeArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from cv_bridge import CvBridge

BEST_EFFORT_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)

class ZEDNode(Node):

    def __init__(self):
        super().__init__('stereo_predictor')

        # initialize all publishers
        self.left_publisher = self.create_publisher(msg_type=Image,
                                                     topic=LEFT_IMAGE_TOPIC,
                                                     qos_profile=BEST_EFFORT_QOS_PROFILE)
        # self.right_publisher = self.create_publisher(msg_type=Image,
        #                                              topic=RIGHT_IMAGE_TOPIC,
        #                                              qos_profile=BEST_EFFORT_QOS_PROFILE)
        # self.depth_publisher = self.create_publisher(msg_type=Image,
        #                                              topic=DEPTH_IMAGE_TOPIC,
        #                                              qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.xyz_publisher = self.create_publisher(msg_type=Image,
                                                   topic=XYZ_IMAGE_TOPIC,
                                                   qos_profile=BEST_EFFORT_QOS_PROFILE)

        # initialize timer interval for publishing the data
        frame_rate = 50
        self.data_syncer = self.create_timer(1/frame_rate, self.inference)

        # initialize the ZEDSDK API for receiving raw data
        self.zed = ZEDSDK()
        self.zed.open()

        self.bridge = CvBridge()
        self.frame_id = 0

    def inference(self):
        # try displaying the image

        s = time.time()

        left, right, depth, xyz = self.zed.grab_data()
        
        # convert the data and check that it is the same going and backwards
        # have to extract out nan values that don't count to compare image values
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
        print(f"Publishing data: {1000 * (t - s):.3f}ms")

        

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = ZEDNode()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()