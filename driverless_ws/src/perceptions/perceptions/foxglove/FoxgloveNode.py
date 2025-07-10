import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from std_msgs.msg import Float64
from geometry_msgs.msg import Point, TwistStamped
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from interfaces.msg import ConeArray, SplineFrames, PPMConeArray
from perceptions.topics import POINT_TOPIC, POINT_2_TOPIC
import math
import argparse
import sys
import struct

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

        # Publishers and Subscribers
        # Replace MarkerArray publishers with PointCloud2 publishers
        self.cone_publisher = self.create_publisher(
            PointCloud2, 'cone_points', 
            qos_profile=BEST_EFFORT_QOS_PROFILE
        )
        self.colored_cone_publisher = self.create_publisher(
            PointCloud2, 'colored_cone_points', 
            qos_profile=BEST_EFFORT_QOS_PROFILE
        )
        self.associated_cone_publisher = self.create_publisher(
            PointCloud2, 'associated_cone_points', 
            qos_profile=BEST_EFFORT_QOS_PROFILE
        )

        self.spline_publisher = self.create_publisher(
            MarkerArray, 'spline_markers', 
            qos_profile=BEST_EFFORT_QOS_PROFILE
        )

        self.speed_publisher = self.create_publisher(
            Float64, 'speed', 
            qos_profile=BEST_EFFORT_QOS_PROFILE
        )

        self.cone_subscriber = self.create_subscription(
            ConeArray, '/perc_cones', 
            self.colored_cone_array_callback, 
            qos_profile=BEST_EFFORT_QOS_PROFILE
        )

        self.cone_subscriber = self.create_subscription(
            ConeArray, '/associated_perc_cones', 
            self.associated_cone_array_callback, 
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

        self.speed_subscriber = self.create_subscription(
            TwistStamped, '/filter/twist', 
            self.speed_callback, 
            qos_profile=BEST_EFFORT_QOS_PROFILE
        )

    def cone_array_callback(self, msg: PPMConeArray):
        # Create point cloud message for cones
        point_cloud = self.create_point_cloud(msg.header.stamp, 'hesai_lidar')
        
        points = []
        for cone_points in msg.cone_array:
            cone = cone_points.cone_points[0]
            # x, y, z, rgb (packed as float)
            # Default color: gray (128, 128, 128)
            points.append([cone.x, cone.y, 0.0, self.pack_rgb(128, 128, 128)])
        
        if points:
            point_cloud.data = self.pack_points(points)
            point_cloud.width = len(points)
        
        self.cone_publisher.publish(point_cloud)
    
    def colored_cone_array_callback(self, msg: ConeArray):
        if self.print_counts:
            print(
                f"{len(msg.blue_cones):<3} Blue Cones | "
                f"{len(msg.yellow_cones):<3} Yellow Cones | "
                f"{len(msg.orange_cones):<3} Orange Cones | "
                f"{len(msg.big_orange_cones):<3} Big Orange Cones | "
                f"{len(msg.unknown_color_cones):<3} Unknown Color Cones"
            )
        point_cloud = self.create_colored_point_cloud(msg)
        self.colored_cone_publisher.publish(point_cloud)

    def associated_cone_array_callback(self, msg: ConeArray):
        if self.print_counts:
            print(
                f"{len(msg.blue_cones):<3} Blue Cones | "
                f"{len(msg.yellow_cones):<3} Yellow Cones | "
                f"{len(msg.orange_cones):<3} Orange Cones | "
                f"{len(msg.big_orange_cones):<3} Big Orange Cones | " f"{len(msg.unknown_color_cones):<3} Unknown Color Cones"
            )
        point_cloud = self.create_colored_point_cloud(msg)
        self.associated_cone_publisher.publish(point_cloud)
    
    def create_point_cloud(self, stamp, frame_id):
        # Create a new PointCloud2 message
        point_cloud = PointCloud2()
        point_cloud.header.stamp = stamp
        point_cloud.header.frame_id = frame_id
        point_cloud.height = 1
        point_cloud.width = 0  # Will set this later based on number of points
        
        # Define fields for x, y, z, and rgb (packed as float)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        point_cloud.fields = fields
        
        point_cloud.is_bigendian = False
        point_cloud.point_step = 16  # 4 fields * 4 bytes each (float32)
        point_cloud.row_step = 0  # Will set this later
        point_cloud.is_dense = True
        
        return point_cloud
    
    def pack_rgb(self, r, g, b):
        # Pack RGB values into a single float
        rgb = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]
        return rgb
    
    def pack_points(self, points):
        # Pack points into a byte array
        points_data = bytearray()
        for point in points:
            points_data.extend(struct.pack('ffff', *point))
        
        return bytes(points_data)
    
    def create_colored_point_cloud(self, msg: ConeArray):
        point_cloud = self.create_point_cloud(msg.header.stamp, 'hesai_lidar')
        
        points = []
        
        # Add blue cones: RGB(0, 0, 255)
        for cone in msg.blue_cones:
            points.append([cone.x, cone.y, 0.0, self.pack_rgb(0, 0, 255)])
            
        # Add yellow cones: RGB(255, 255, 0)
        for cone in msg.yellow_cones:
            points.append([cone.x, cone.y, 0.0, self.pack_rgb(255, 255, 0)])
            
        # Add orange cones: RGB(255, 165, 0)
        for cone in msg.orange_cones:
            points.append([cone.x, cone.y, 0.0, self.pack_rgb(255, 165, 0)])
            
        # Add big orange cones: RGB(255, 69, 0)
        for cone in msg.big_orange_cones:
            points.append([cone.x, cone.y, 0.0, self.pack_rgb(255, 69, 0)])
            
        # Add unknown cones: RGB(128, 128, 128)
        for cone in msg.unknown_color_cones:
            points.append([cone.x, cone.y, 0.0, self.pack_rgb(128, 128, 128)])
        
        if points:
            point_cloud.data = self.pack_points(points)
            point_cloud.width = len(points)
            point_cloud.row_step = point_cloud.width * point_cloud.point_step
        
        return point_cloud

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
    
    def speed_callback(self, msg: TwistStamped):
        speed_msg = Float64()
        speed_msg.data = math.sqrt(msg.twist.linear.x ** 2 + msg.twist.linear.y ** 2)
        self.speed_publisher.publish(speed_msg)


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