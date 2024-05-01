# ROS2 imports
import rclpy

# for subscribing to sensor data
from perceptions.ros.utils.PredictNode import PredictNode

# for doing prediction on sensor data
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

# ROS2 message types
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TwistStamped, QuaternionStamped
from interfaces.msg import ConeArray

# ROS2 msg to python datatype conversions
import perceptions.ros.utils.conversions as conv
from perceptions.topics import POINT_TOPIC, TWIST_TOPIC, QUAT_TOPIC, PERC_CONE_TOPIC

# perceptions Library visualization functions (for 3D data)
import perc22a.predictors.utils.lidar.visualization as vis
from perc22a.data.utils.DataType import DataType
from perc22a.data.utils.DataInstance import DataInstance

# Cone Merger and pipeline enum type
from perc22a.mergers.MergerInterface import Merger
from perc22a.mergers.PipelineType import PipelineType
from perc22a.mergers.merger_factory import \
    create_lidar_merger, \
    create_zed_merger, \
    create_all_merger, \
    create_any_merger

from interfaces.msg import SplineFrames
from geometry_msgs.msg import Point

import perceptions.ros.utils.conversions as conv
from perc22a.svm.SVM import SVM
from perc22a.predictors.utils.vis.Vis2D import Vis2D
from perc22a.utils.Timer import Timer

from perc22a.predictors.utils.ConeState import ConeState

import time

NODE_NAME = "end_to_end_node"

# configure QOS profile
BEST_EFFORT_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 1)
RELIABLE_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.RELIABLE,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 1)

class EndToEndNode(Node):

    def __init__(self):
        super().__init__(NODE_NAME)

        # data subscribers
        self.point_subscriber = self.create_subscription(PointCloud2, POINT_TOPIC, self.points_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.twist_subscriber = self.create_subscription(TwistStamped, TWIST_TOPIC, self.twist_callback, qos_profile=RELIABLE_QOS_PROFILE)
        self.quat_subscriber = self.create_subscription(QuaternionStamped, QUAT_TOPIC, self.quat_callback, qos_profile=RELIABLE_QOS_PROFILE)

        # publishers
        self.midline_pub = self.create_publisher(msg_type=SplineFrames,
                                                 topic="/spline",
                                                 qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.cone_pub = self.create_publisher(msg_type=ConeArray,
                                              topic=PERC_CONE_TOPIC,
                                              qos_profile=BEST_EFFORT_QOS_PROFILE)

        # parts of the pipeline 
        self.predictor = self.init_predictor()
        self.merger = create_lidar_merger()
        self.cone_state = ConeState()
        self.svm = SVM()

        # attributes for storing twist and quaternion for motion modeling
        self.curr_twist = None
        self.curr_quat = None

        # debugging utilities
        self.vis = Vis2D()
        self.timer = Timer()
        return
    
    def init_predictor(self):
        return LidarPredictor()
    
    def twist_callback(self, curr_twist):
        self.curr_twist = curr_twist

    def quat_callback(self, curr_quat):
        self.curr_quat = curr_quat
    
    def points_callback(self, msg):
        # if doesn't have GPS data, return
        if self.curr_twist is None or self.curr_quat is None:
            self.get_logger().warn(f"No twist or quat data. Turn on GPS!")
            return

        # initialize time for staleness of data 
        data_time = self.get_clock().now()

        # convert pointcloud message into numpy array and get MotionInfo
        data = {}
        data[DataType.HESAI_POINTCLOUD] = conv.pointcloud2_to_npy(msg)

        mi = conv.gps_to_motion_info(self.curr_twist, self.curr_quat)

        # predict lidar
        self.timer.start("lidar")
        cones = self.predictor.predict(data)
        time_lidar = self.timer.end("lidar", ret=True)

        # update using cone merger
        self.timer.start("merge+color+state")
        self.merger.add(cones, PipelineType.LIDAR)
        cones = self.merger.merge()
        self.merger.reset()

        # recolor using SVM
        cones = self.svm.recolor(cones)

        # update overall cone state
        # TODO: should separately update cones and then return cones relevant for svm
        cones = self.cone_state.update(cones, mi)
        time_state = self.timer.end("merge+color+state", ret=True)

        # spline
        self.timer.start("spline")
        downsampled_boundary_points = self.svm.cones_to_midline(cones)
        time_spline = self.timer.end("spline", ret=True)

        # convert spline points to ROS2 SplineFrame message
        points = []
        msg = SplineFrames()

        for np_point in downsampled_boundary_points:
            new_point = Point()
            new_point.x = float(np_point[0]) # turning into not SAE coordinates
            new_point.y = float(np_point[1])
            new_point.z = float(0)
            points.append(new_point)

        self.vis.set_cones(cones)
        if len(downsampled_boundary_points) > 0:
            self.vis.set_points(downsampled_boundary_points)
        self.vis.update()

        if len(points) < 2:
            print(f"LESS THAN 2 FRAMES {len(cones)}")
            return
        
        msg.frames = points
        msg.orig_data_stamp = data_time.to_msg()
        self.midline_pub.publish(msg)

        # convert cones to ConeArray ROS2 message and publish
        self.timer.start("cone_pub")
        cone_msg = conv.cones_to_msg(cones)
        self.cone_pub.publish(cone_msg)
        time_cone_pub = self.timer.end("cone_pub", ret=True)

        # done publishing spline
        curr_time = self.get_clock().now()
        # print the timings of everything
        print(f"{len(cones):<3} cones {len(downsampled_boundary_points):<3} frames {(curr_time.nanoseconds - data_time.nanoseconds) / 1000000:.3f}ms lidar: {time_lidar:.1f}ms merge+color+state: {time_state:.1f}ms spline: {time_spline:.1f}ms cone_pub: {time_cone_pub:.1f}ms")

        return


def main(args=None):
    rclpy.init(args=args)

    stereo_node = EndToEndNode()

    rclpy.spin(stereo_node)

    stereo_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()