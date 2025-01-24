import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from perceptions.topics import POINT_TOPIC, POINT_2_TOPIC

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from interfaces.msg import ConeArray, SplineFrames
import numpy as np
import argparse
import sys

NODE_NAME = 'foxglove_node'

BEST_EFFORT_QOS_PROFILE = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.VOLATILE,
    depth=1
)

class FoxgloveNode(Node):
    def __init__(self, print_counts):
        super().__init__(NODE_NAME)

        self.print_counts = print_counts
        self.last_marker_count = 0
        self.last_spline_marker_count = 0

        # Subscribers and Publishers
        self.cone_publisher = self.create_publisher(MarkerArray, 'cone_markers', qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.spline_publisher = self.create_publisher(MarkerArray, 'spline_markers', qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.fused_publisher = self.create_publisher(PointCloud2, '/fused_lidar_points', qos_profile=BEST_EFFORT_QOS_PROFILE)

        self.cone_subscriber = self.create_subscription(ConeArray, '/perc_cones', self.cone_array_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.spline_subscriber = self.create_subscription(SplineFrames, '/spline', self.spline_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.point_1_subscriber = self.create_subscription(PointCloud2, POINT_TOPIC, self.points_callback_1, qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.point_2_subscriber = self.create_subscription(PointCloud2, POINT_2_TOPIC, self.points_callback_2, qos_profile=BEST_EFFORT_QOS_PROFILE)

        # LiDAR points and transformation matrices
        self.points_1 = None
        self.points_2 = None

        self.tf_mat_left = np.array([
            [0.76604444, -0.64278764, 0., -0.18901],
            [0.64278764, 0.76604444, 0., 0.15407],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])
        self.tf_mat_right = np.array([
            [0.76604444, 0.64278764, 0., -0.16541],
            [-0.64278764, 0.76604444, 0., -0.12595],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])

    def points_callback_1(self, msg: PointCloud2):
        self.points_1 = self.transform_points(msg, self.tf_mat_left)
        self.publish_fused_points()

    def points_callback_2(self, msg: PointCloud2):
        self.points_2 = self.transform_points(msg, self.tf_mat_right)
        self.publish_fused_points()

    def transform_points(self, msg, tf_mat):
        points = list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
        if not points:
            return np.array([])
        
        points = np.array([[p[0], p[1], p[2], p[3]] for p in points][::2])
        points_h = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
        transformed = np.dot(points_h, tf_mat.T)[:, :3]
        
        return np.hstack((transformed, points[:, 3:4]))

    def publish_fused_points(self):
        if self.points_1 is not None and self.points_2 is not None:
            fused_points = np.vstack((self.points_1, self.points_2))
            header = PointCloud2().header
            header.frame_id = 'hesai_lidar'
            header.stamp = self.get_clock().now().to_msg()
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
            ]
            pc2_msg = pc2.create_cloud(header, fields, fused_points.tolist())
            self.fused_publisher.publish(pc2_msg)
            self.get_logger().info(f"Published {fused_points.shape[0]} fused LiDAR points")

    def cone_array_callback(self, msg: ConeArray):
        if self.print_counts:
            print(
                f"{len(msg.blue_cones):<3} Blue Cones | "
                f"{len(msg.yellow_cones):<3} Yellow Cones | "
                f"{len(msg.orange_cones):<3} Orange Cones | "
                f"{len(msg.big_orange_cones):<3} Big Orange Cones | "
                f"{len(msg.unknown_color_cones):<3} Unknown Color Cones"
            )
        marker_array = self.create_marker_array(msg)
        self.cone_publisher.publish(marker_array)

    def create_marker_array(self, msg: ConeArray):
        marker_array = MarkerArray()
        marker_id = 0
        namespace = "cone_markers"

        def add_cones(cones, color):
            nonlocal marker_id
            for cone in cones:
                marker = Marker()
                marker.header.frame_id = 'hesai_lidar'
                marker.header.stamp = msg.header.stamp
                marker.ns = namespace
                marker.id = marker_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = cone.y
                marker.pose.position.y = -cone.x
                marker.pose.position.z = -0.7
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.2
                marker.color.r = color[0] / 255.0
                marker.color.g = color[1] / 255.0
                marker.color.b = color[2] / 255.0
                marker.color.a = 1.0
                marker_array.markers.append(marker)
                marker_id += 1

        add_cones(msg.blue_cones, [0, 0, 255])
        add_cones(msg.yellow_cones, [255, 255, 0])
        add_cones(msg.orange_cones, [255, 165, 0])
        add_cones(msg.big_orange_cones, [255, 69, 0])
        add_cones(msg.unknown_color_cones, [128, 128, 128])

        for old_id in range(marker_id, self.last_marker_count):
            delete_marker = Marker()
            delete_marker.header.frame_id = 'hesai_lidar'
            delete_marker.header.stamp = msg.header.stamp
            delete_marker.ns = namespace
            delete_marker.id = old_id
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        self.last_marker_count = marker_id
        return marker_array

    def spline_callback(self, msg: SplineFrames):
        marker_array = MarkerArray()

        spline_marker = Marker()
        spline_marker.header.frame_id = 'hesai_lidar'
        spline_marker.header.stamp = msg.header.stamp
        spline_marker.ns = 'spline'
        spline_marker.id = 0
        spline_marker.type = Marker.LINE_STRIP
        spline_marker.action = Marker.ADD
        spline_marker.scale.x = 0.1
        spline_marker.color.r = 0.0
        spline_marker.color.g = 1.0
        spline_marker.color.b = 0.0
        spline_marker.color.a = 1.0

        transformed_points = []
        for frame in msg.frames:
            point = Point()
            point.x = frame.y
            if point.x < 1:
                continue
            point.y = -frame.x
            point.z = -0.7
            transformed_points.append(point)

        spline_marker.points = transformed_points
        marker_array.markers.append(spline_marker)

        car_marker = Marker()
        car_marker.header.frame_id = 'hesai_lidar'
        car_marker.header.stamp = msg.header.stamp
        car_marker.ns = 'car'
        car_marker.id = 1
        car_marker.type = Marker.CUBE
        car_marker.action = Marker.ADD
        car_marker.pose.position.x = 0.0
        car_marker.pose.position.y = 0.0
        car_marker.pose.position.z = -0.7
        car_marker.pose.orientation.w = 1.0
        car_marker.scale.x = 1.8
        car_marker.scale.y = 1.2
        car_marker.scale.z = 0.5
        car_marker.color.r = 1.0
        car_marker.color.g = 0.0
        car_marker.color.b = 0.0
        car_marker.color.a = 1.0
        marker_array.markers.append(car_marker)

        self.spline_publisher.publish(marker_array)


def main(args=None):
    parser = argparse.ArgumentParser(description="Foxglove Nod")
    parser.add_argument('-p', '--print', action='store_true', help="Print cone counts")
    parsed_args = parser.parse_args(args=sys.argv[1:])

    rclpy.init(args=args)
    node = FoxgloveNode(print_counts=parsed_args.print)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
