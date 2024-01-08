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

class AggregateNode(Node):

    def __init__(self):
        super().__init__('aggregate_node')
        self.left_color_subscriber = self.create_subscription(Image, LEFT_IMAGE_TOPIC, self.left_color_callback, qos_profile=RELIABLE_QOS_PROFILE)
        self.pointcloud_subscriber = self.create_subscription(PointCloud2, POINT_TOPIC, self.pointcloud_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.img_publisher = self.create_publisher(Image, "/projection_img", RELIABLE_QOS_PROFILE)

        self.bridge = CvBridge()

        self.hz = 2
        self.timer = self.create_timer(1 / self.hz, self.timer_callback)
        self.count = 0

        # # initialize the ZEDSDK API for receiving raw data
        # self.zed = ZEDSDK()
        # self.zed.open()
        
        self.latest_img = None
        self.latest_points = None
    
    def left_color_callback(self, img_msg):
        self.latest_img = img_msg
    
    def pointcloud_callback(self, points_msg):
        self.latest_points = points_msg
        self.count += 1
    
    def timer_callback(self):
        print(f"Count: {self.count}")
        self.count = 0
        if self.latest_img != None and self.latest_points != None:
            img = conv.img_to_npy(self.latest_img)
            points = conv.pointcloud2_to_npy(self.latest_points)[:, :3] # Nx3

            points = points[:, [1, 2, 0]]
            points[:,0] = -points[:,0]
            points[:,1] = -points[:,1]

            points = points[np.any(points[:, :2] != 0, axis=1)]

            intrinsic = np.asarray([
                [686.83996582, 0, 676.8425293],
                [0, 686.83996582, 369.63140869],
                [0, 0, 1]
            ])

            # print(f"x (right positive): {min(points[:, 0])}, {max(points[:, 0])}")
            # print(f"y (down positive): {min(points[:, 1])}, {max(points[:, 1])}")
            # print(f"z (forward positive): {min(points[:, 2])}, {max(points[:, 2])}")
            
            points[:, 1] += 0.029 # Z-axis translation from lidar to stereo, now in camera frame
            points[:, 0] -= 0.25 # Z-axis translation from lidar to stereo, now in camera frame

            img_coords = intrinsic @ points.T
            img_coords[:2, :] /= img_coords[2]

            mask = img_coords[2, :]  < 2
            img_coords = img_coords[:, mask]

            # x_idxs = np.where((img_coords[0, :] >= 0) & (img_coords[0, :] < img.shape[0]))
            # y_idxs = np.where((img_coords[1, :] >= 0) & (img_coords[1, :] < img.shape[1]))
            # common_idxs = np.intersect1d(x_idxs, y_idxs)
            # valid_pts = img_coords[:, common_idxs]
            min_depth = min(img_coords[2, :])
            max_depth = max(img_coords[2, :])
            for i in range(img_coords.shape[1]):
                # u, v, depth = valid_pts[i, :]
                u, v, z = img_coords[:, i]
                norm_z = (z - min_depth) / (max_depth - min_depth) # between 0, 1
                radius = 4 - int(4 * norm_z)
                color = (0, 0, int(norm_z * 255))
                u = int(u)
                v = int(v)
                cv2.circle(img, (u, v), radius=radius, color=color, thickness=-1)
            img_msg = self.bridge.cv2_to_imgmsg(img)
            self.img_publisher.publish(img_msg)

        else:
            print("not got data")

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = AggregateNode()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()