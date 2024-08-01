import rclpy
from rclpy.node import Node
from perceptions.topics import GPS_TOPIC, POINT_TOPIC, POINT_TOPIC_ADJ
from sensor_msgs.msg import PointCloud2, NavSatFix
from interfaces.msg import PointCloud2TimeAdj

import time


class TimeAdjustNode(Node):
    def __init__(self):
        super().__init__('time_adjust_node')

        lidar_topic = POINT_TOPIC
        gnss_topic = GPS_TOPIC


        self.got_first_gnss = False
        self.got_first_lidar = False
        self.got_diff = False
        #1.) get the time difference between the 1st msg of both topics
        self.gnss_subscriber = self.create_subscription(
                                NavSatFix,
                                gnss_topic,
                                self.gnss_callback,
                                10)
        self.lidar_subscriber = self.create_subscription(
                                PointCloud2,
                                lidar_topic,
                                self.lidar_callback,
                                10)

        self.lidar_publisher = self.create_publisher(
                                PointCloud2TimeAdj,
                                POINT_TOPIC_ADJ,
                                10)
        self.gnss_subscriber
        self.lidar_subscriber

    def gnss_callback(self, msg):
        if not self.got_first_gnss:
            self.first_gnss_sec = msg.header.stamp.sec
            self.first_gnss_nanosec = msg.header.stamp.nanosec
            self.got_first_gnss = True


    def lidar_callback(self, msg):
        self.get_logger().info("time_adjust_node: lidar callback")
        if not self.got_first_lidar:
            self.first_lidar_sec = msg.header.stamp.sec
            self.first_lidar_nanosec = msg.header.stamp.nanosec

            self.got_first_lidar = True

        if self.got_first_gnss and self.got_first_gnss and (not self.got_diff):
                self.diff_sec = self.first_gnss_sec - self.first_lidar_sec + 1
                # if self.first_lidar_nanosec > self.first_gnss_nanosec:
                #     self.diff_sec = self.diff_sec - 1

                self.got_diff = True

        if self.got_diff:
            new_msg = PointCloud2TimeAdj()
            new_msg.point_cloud = msg
            new_msg.header.stamp.sec = msg.header.stamp.sec + self.diff_sec
            new_msg.header.stamp.nanosec = msg.header.stamp.nanosec
            self.lidar_publisher.publish(new_msg)




def main(args=None):
    rclpy.init(args=args)
    node = TimeAdjustNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
