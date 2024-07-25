import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from perceptions.topics import LEFT_IMAGE_TOPIC, LEFT2_IMAGE_TOPIC
from perceptions.topics import XYZ_IMAGE_TOPIC, XYZ2_IMAGE_TOPIC
from perceptions.topics import POINT_TOPIC
from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
from geometry_msgs.msg import TwistStamped, QuaternionStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from interfaces.msg import SyncedLidarOdom

import time

class SyncNode(Node):
    def __init__(self):
        super().__init__('sync_node')
        
        # Define your topic names
        zed_topic = LEFT_IMAGE_TOPIC
        zed2_topic = LEFT2_IMAGE_TOPIC
        zed_xyz_topic = XYZ_IMAGE_TOPIC
        zed_xyz2_topic = XYZ2_IMAGE_TOPIC

        lidar_topic = POINT_TOPIC

        gnss_topic = '/gnss'
        imu_linear_velocity_topic = '/filter/twist'
        imu_orientation_topic = '/filter/quaternion'

        self.get_logger().info('Starting sync_node')

        # Create subscribers for each topic
        # zed_sub = Subscriber(self, Image, zed_topic)
        # zed2_sub = Subscriber(self, Image, zed2_topic)
        # xyz_sub  = Subscriber(self, Image, zed_xyz_topic)
        # xyz2_sub = Subscriber(self, Image, zed_xyz2_topic)

        self.lidar_sub = Subscriber(self, PointCloud2, lidar_topic)
    
        self.gnss_sub = Subscriber(self, NavSatFix, gnss_topic)
        self.imu_linear_velocity_sub = Subscriber(self, TwistStamped, imu_linear_velocity_topic)
        self.imu_orientation_sub = Subscriber(self, QuaternionStamped, imu_orientation_topic)

        # Set queue size based on your requirements
        queue_size = 1000

        # Use ApproximateTimeSynchronizer to synchronize messages
        self.synced_sub = ApproximateTimeSynchronizer([
                                            self.lidar_sub,
                                            self.gnss_sub,
                                            self.imu_linear_velocity_sub,
                                            self.imu_orientation_sub
                                            ], 
                                            queue_size=queue_size, slop=2)
        self.synced_sub.registerCallback(self.sync_callback)

        # Create a publisher for your DataFrame message
        self.synced_pub = self.create_publisher(SyncedLidarOdom, '/synced_data', 5)
        self.get_logger().info('Init sync_node complete')

    def sync_callback(self, 
                        point_cloud_msg,
                        gnss_msg,
                        linvelocity_msg,
                        orientation_msg
                        ):
        s = time.time()

        # Your callback logic here
        self.get_logger().info("Received synchronized messages")
        self.get_logger().info('Received pointcloud, gps, linear velocity, orientation')
        # Create a YourDataFrame message and populate its fields
        synced_msg = SyncedLidarOdom()
        # synced_msg.left_color = image_msg
        # synced_msg.left2_color = image2_msg
        # synced_msg.xyz_image = xyz_msg
        # synced_msg.xyz2_image = xyz2_msg
        # synced_msg.point_cloud = point_cloud_msg
        
        synced_msg.gps_data = gnss_msg
        synced_msg.velocity = linvelocity_msg
        synced_msg.orientation = orientation_msg

        # Publish the combined DataFrame message
        self.synced_pub.publish(synced_msg)

        e = time.time()
        print(f"sync publish: {int(1000 * (e - s))}ms")

def main(args=None):
    rclpy.init(args=args)
    node = SyncNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
