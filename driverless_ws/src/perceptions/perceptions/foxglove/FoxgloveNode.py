import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from interfaces.msg import ConeArray, SplineFrames
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
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
        self.last_marker_count = 0  # Track the number of cone markers published previously
        self.last_spline_marker_count = 0  # Track number of spline markers
        
        self.cone_publisher = self.create_publisher(
            MarkerArray,
            'cone_markers',
            10
        )

        self.spline_publisher = self.create_publisher(
            MarkerArray,
            'spline_markers',
            10
        )
        
        self.subscriber = self.create_subscription(
            ConeArray,
            '/perc_cones',
            self.cone_array_callback,
            qos_profile=BEST_EFFORT_QOS_PROFILE
        )

        self.spline_subscriber = self.create_subscription(
            SplineFrames,
            '/spline',
            self.spline_callback,
            qos_profile=BEST_EFFORT_QOS_PROFILE
        )

    def cone_array_callback(self, msg: ConeArray):
        # Print cone counts if the flag is enabled
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

        # Delete leftover markers if any
        for old_id in range(marker_id, self.last_marker_count):
            delete_marker = Marker()
            delete_marker.header.frame_id = 'hesai_lidar'
            delete_marker.header.stamp = msg.header.stamp
            delete_marker.ns = namespace
            delete_marker.id = old_id
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        # Update the last marker count
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
        spline_marker.scale.x = 0.1  # Line width
        spline_marker.color.r = 0.0
        spline_marker.color.g = 1.0
        spline_marker.color.b = 0.0
        spline_marker.color.a = 1.0
        
        # Transform spline points to the correct orientatation
        transformed_points = []
        for frame in msg.frames:
            transformed_point = Point()
            transformed_point.x = frame.y
            # Crops spline so that it only shows the part in front of the car
            if transformed_point.x < 1:
                continue
            transformed_point.y = -frame.x
            transformed_point.z = -0.7
            transformed_points.append(transformed_point)
        spline_marker.points = transformed_points
        marker_array.markers.append(spline_marker)

        # Car Marker
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
    parser = argparse.ArgumentParser(description="Foxglove Cone Visualization Node")
    parser.add_argument('-p', '--print', action='store_true', help="Print cone counts to the console")
    parsed_args = parser.parse_args(args=sys.argv[1:])

    rclpy.init(args=args)
    node = FoxgloveNode(print_counts=parsed_args.print)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
