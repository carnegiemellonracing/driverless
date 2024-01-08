''' ZEDNode.py

File contains implementation of publisher for stereocamera images and depth
maps for the purpose of perceptinos algorithms. Raw data is made accessible
via the ZEDSDK API which is then marshalled and published to the topics
listed out below
'''

import rclpy
from rclpy.node import Node
import cv2
import torch
import numpy as np
import time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import perceptions.ros.utils.conversions as conv

from perceptions.ros.utils.topics import LEFT_IMAGE_TOPIC, RIGHT_IMAGE_TOPIC, XYZ_IMAGE_TOPIC, DEPTH_IMAGE_TOPIC, POINT_TOPIC
from perceptions.ros.utils.zed import ZEDSDK

from eufs_msgs.msg import ConeArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header

from cv_bridge import CvBridge

RELIABLE_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.RELIABLE,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 1)

BEST_EFFORT_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 1)

class LidarTestNode(Node):

    def __init__(self):
        super().__init__('lidar_test_node')
        self.pointcloud_subscriber = self.create_subscription(PointCloud2, POINT_TOPIC, self.pointcloud_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.pointcloud_publisher = self.create_publisher(PointCloud2, "/republished", BEST_EFFORT_QOS_PROFILE)
        self.image_publisher = self.create_publisher(Image, "/img", BEST_EFFORT_QOS_PROFILE)

        self.bridge = CvBridge()
        self.img = self.bridge.cv2_to_imgmsg(np.random.randint(0, 256, size=(640, 480, 3), dtype=np.uint8))

        self.hz = 10
        self.timer = self.create_timer(1 / self.hz, self.timer_callback)
        self.count = 0
        self.latest_points = None
    
    def pointcloud_callback(self, points_msg):
        self.latest_points = points_msg
        self.count += 1

    def timer_callback(self):
        print(f"Count: {self.count}")
        self.count = 0

        self.pointcloud_publisher.publish(self.latest_points) 
        self.image_publisher.publish(self.img)

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = LidarTestNode()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()