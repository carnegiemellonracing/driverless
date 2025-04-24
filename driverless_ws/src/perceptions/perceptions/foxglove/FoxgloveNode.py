import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, Point
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from interfaces.msg import ConeArray, SplineFrames, PPMConeArray, PPMConePoints
from perceptions.topics import POINT_TOPIC, POINT_2_TOPIC
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

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Transformation matrices
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

        # Publishers and Subscribers
        self.cone_publisher = self.create_publisher(
            MarkerArray, 'cone_markers', 
            qos_profile=BEST_EFFORT_QOS_PROFILE
        )
        self.colored_cone_publisher = self.create_publisher(
            MarkerArray, 'colored_cone_markers', 
            qos_profile=BEST_EFFORT_QOS_PROFILE
        )
        self.spline_publisher = self.create_publisher(
            MarkerArray, 'spline_markers', 
            qos_profile=BEST_EFFORT_QOS_PROFILE
        )

        self.cone_subscriber = self.create_subscription(
            ConeArray, '/perc_cones', 
            self.colored_cone_array_callback, 
            qos_profile=BEST_EFFORT_QOS_PROFILE
        )
        self.cone_subscriber = self.create_subscription(
            PPMConeArray, '/cpp_cones', 
            self.cone_array_callback, 
            qos_profile=BEST_EFFORT_QOS_PROFILE
        )
        self.spline_subscriber = self.create_subscription(
            SplineFrames, '/spline', 
            self.spline_callback, 
            qos_profile=BEST_EFFORT_QOS_PROFILE
        )
        self.point_1_subscriber = self.create_subscription(
            PointCloud2, POINT_TOPIC, 
            self.points_callback_1, 
            qos_profile=BEST_EFFORT_QOS_PROFILE
        )
        self.point_2_subscriber = self.create_subscription(
            PointCloud2, POINT_2_TOPIC,
            self.points_callback_2,
            qos_profile=BEST_EFFORT_QOS_PROFILE
        )

    def points_callback_1(self, msg: PointCloud2):
        self.publish_tf(self.tf_mat_left, 'dual_lidar', 'hesai_lidar')

    def points_callback_2(self, msg: PointCloud2):
        self.publish_tf(self.tf_mat_right, 'dual_lidar', 'hesai_lidar2')

    def publish_tf(self, tf_mat, parent_frame, child_frame):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        # Extract translation directly from the transformation matrix
        t.transform.translation.x = tf_mat[0, 3]
        t.transform.translation.y = tf_mat[1, 3]
        t.transform.translation.z = tf_mat[2, 3]

        # Extract rotation matrix and convert to quaternion
        rotation_matrix = tf_mat[:3, :3]
        q = self.rotation_matrix_to_quaternion(rotation_matrix)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    def rotation_matrix_to_quaternion(self, R):
        # Convert rotation matrix to quaternion
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            S = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / S
            x = (R[2, 1] - R[1, 2]) * S
            y = (R[0, 2] - R[2, 0]) * S
            z = (R[1, 0] - R[0, 1]) * S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        return [x, y, z, w]

    def cone_array_callback(self, msg: PPMConeArray):
        marker_array = MarkerArray()
        marker_id = 0
        for cone_points in msg.cone_array:
            cone = cone_points.cone_points[0]
            marker = Marker()
            marker.header.frame_id = 'hesai_lidar'
            marker.header.stamp = msg.header.stamp
            marker.ns = "cone_markers"
            marker.id = marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = cone.x
            marker.pose.position.y = cone.y
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker_array.markers.append(marker)
            marker_id += 1
        self.cone_publisher.publish(marker_array)
    
    def colored_cone_array_callback(self, msg: ConeArray):
        if self.print_counts:
            print(
                f"{len(msg.blue_cones):<3} Blue Cones | "
                f"{len(msg.yellow_cones):<3} Yellow Cones | "
                f"{len(msg.orange_cones):<3} Orange Cones | "
                f"{len(msg.big_orange_cones):<3} Big Orange Cones | "
                f"{len(msg.unknown_color_cones):<3} Unknown Color Cones"
            )
        marker_array = self.create_marker_array(msg)
        self.colored_cone_publisher.publish(marker_array)

    def create_marker_array(self, msg: ConeArray):
        marker_array = MarkerArray()
        marker_id = 0
        namespace = "colored_cone_markers"

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
                marker.pose.position.x = cone.x
                marker.pose.position.y = cone.y
                marker.pose.position.z = 0.0
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
            point.x = frame.x
            if point.x < 1:
                continue
            point.y = frame.y
            point.z = 0.0
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
    parser = argparse.ArgumentParser(description="Foxglove Node")
    parser.add_argument('-p', '--print', action='store_true', help="Print cone counts")
    parsed_args = parser.parse_args(args=sys.argv[1:])

    rclpy.init(args=args)
    node = FoxgloveNode(print_counts=parsed_args.print)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()