import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from eufs_msgs import DataFrame

class SyncNode(Node):
    def __init__(self):
        super().__init__('sync_node')

        # Define your topic names
        zed_topic = '/zedsdk_left_color_image'
        lidar_topic = '/lidar_points'

        # Create subscribers for each topic
        zed_sub = Subscriber(self, Image, zed_topic)
        lidar_sub = Subscriber(self, PointCloud2, lidar_topic)

        # Set queue size based on your requirements
        queue_size = 100

        # Use ApproximateTimeSynchronizer to synchronize messages
        sync = ApproximateTimeSynchronizer([zed_sub, lidar_sub], queue_size=queue_size, slop=0.1)
        sync.registerCallback(self.callback)

        # Create a publisher for your DataFrame message
        self.df_pub = self.create_publisher(DataFrame, '/DataFrame', 50)

    def callback(self, image_msg, pointcloud_msg):
        # Your callback logic here
        self.get_logger().info("Received synchronized messages")

        # Convert Image and PointCloud2 messages to DataFrame
        # Replace the following lines with your actual conversion logic
        image_data = np.array(image_msg.data)
        pointcloud_data = np.array(pointcloud_msg.data)
        # combined_data = pd.concat([pd.DataFrame(image_data), pd.DataFrame(pointcloud_data)], axis=1)

        # Create a YourDataFrame message and populate its fields
        df_msg = DataFrame()
        df_msg.image_msg = image_msg
        df_msg.pointcloud_msg = pointcloud_msg

        # Publish the combined DataFrame message
        self.df_pub.publish(df_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SyncNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()