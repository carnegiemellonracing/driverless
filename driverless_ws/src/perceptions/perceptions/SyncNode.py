import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
from geometry_msgs.msg import Vector3Stamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from eufs_msgs.msg import DataFrame

import time

class SyncNode(Node):
    def __init__(self):
        super().__init__('sync_node')

        # Define your topic names
        zed_topic = '/zedsdk_left_color_image'
        zed_xyz_topic = '/zedsdk_point_cloud_image'
        # lidar_topic = '/lidar_points'
        imu_data_topic = '/imu/data'
        imu_linear_velocity_topic = '/filter/velocity'
        # gnss_topic = '/gnss'

        # Create subscribers for each topic
        zed_sub = Subscriber(self, Image, zed_topic)
        xyz_sub  = Subscriber(self, Image, zed_xyz_topic)
        # lidar_sub = Subscriber(self, PointCloud2, lidar_topic)
        imu_data_sub = Subscriber(self, Imu, imu_data_topic)
        imu_linear_velocity_sub = Subscriber(self, Vector3Stamped, imu_linear_velocity_topic)
        # gnss_sub = Subscriber(self, NavSatFix, gnss_topic)

        # Set queue size based on your requirements
        queue_size = 100

        # Use ApproximateTimeSynchronizer to synchronize messages
        sync = ApproximateTimeSynchronizer([zed_sub, xyz_sub, imu_data_sub, imu_linear_velocity_sub], queue_size=queue_size, slop=0.05)
        sync.registerCallback(self.callback)

        # Create a publisher for your DataFrame message
        self.df_pub = self.create_publisher(DataFrame, '/DataFrame', 50)

    def callback(self, image_msg, xyz_msg, imudata_msg, linvelocity_msg):
        s = time.time()

        # Your callback logic here
        self.get_logger().info("Received synchronized messages")

        # Create a YourDataFrame message and populate its fields
        df_msg = DataFrame()
        df_msg.image_msg = image_msg
        df_msg.xyz_msg = xyz_msg
        df_msg.imu_data = imudata_msg
        df_msg.imu_linear_velocity = linvelocity_msg
        # df_msg.gps_data = gnss_msg

        # Publish the combined DataFrame message
        self.df_pub.publish(df_msg)

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