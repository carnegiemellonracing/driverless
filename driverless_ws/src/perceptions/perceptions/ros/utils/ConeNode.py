# ROS2 imports
import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

# cone datatype for ROS and perc22a
from sensor_msgs.msg import PointCloud2
from interfaces.msg import ConeArray
from perc22a.predictors.utils.cones import Cones
from perc22a.predictors.utils.transform.transform import PoseTransformations
import perceptions.ros.utils.conversions as conv

# Cone Merger and pipeline enum type
from perc22a.mergers.MergerInterface import Merger
from perc22a.mergers.PipelineType import PipelineType
from perc22a.mergers.merger_factory import \
    create_lidar_merger, \
    create_zed_merger, \
    create_all_merger, \
    create_any_merger

from perc22a.utils.Timer import Timer

# perceptions Library visualization functions (for 2D data)
from perc22a.predictors.utils.vis.Vis2D import Vis2D

from perceptions.topics import \
    YOLOV5_ZED_CONE_TOPIC, \
    YOLOV5_ZED2_CONE_TOPIC, \
    LIDAR_CONE_TOPIC, \
    PERC_CONE_TOPIC

# general imports
import numpy as np

# configure QOS profile
BEST_EFFORT_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)
RELIABLE_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.RELIABLE,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)

CONE_NODE_NAME = "cone_node"
PUBLISH_FPS = 20
VIS_UPDATE_FPS = 25
MAX_ZED_CONE_RANGE = 12.5

class ConeNode(Node):

    def __init__(self, merger: Merger, debug=True):
        super().__init__(CONE_NODE_NAME)

        # save our merger for merging cone outputs    
        self.cones = Cones()
        self.merger = merger

        # initialize all cone subscribers   
        self.create_subscription(ConeArray, YOLOV5_ZED_CONE_TOPIC, self.yolov5_zed_cone_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.create_subscription(ConeArray, YOLOV5_ZED2_CONE_TOPIC, self.yolov5_zed2_cone_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.create_subscription(ConeArray, LIDAR_CONE_TOPIC, self.lidar_cone_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)

        # initialize cone publisher
        self.publish_timer = self.create_timer(1/PUBLISH_FPS, self.publish_cones)
        self.cone_publisher = self.create_publisher(ConeArray, PERC_CONE_TOPIC, qos_profile=RELIABLE_QOS_PROFILE)

        # deubgging mode visualizer
        if debug:
            self.vis2D = Vis2D()
            self.display_timer = self.create_timer(1/VIS_UPDATE_FPS, self.update_vis)

        # if debugging, initialize visualizer
        self.debug = debug

        # earliest data timings
        self.data_times = {}

        return

    def update_vis(self):
        # update and interact with vis
        self.vis2D.update()
        return

    def yolov5_zed_cone_callback(self, msg):
        '''receive cones from yolov5_zed_node predictor'''
        self.data_times[PipelineType.ZED_PIPELINE] = Time.from_msg(msg.orig_data_stamp)

        cones = conv.msg_to_cones(msg)
        self.merger.add(cones, PipelineType.ZED_PIPELINE)

        return
    
    def yolov5_zed2_cone_callback(self, msg):
        '''receive cones from yolov5_zed2_node predictor'''
        self.data_times[PipelineType.ZED2_PIPELINE] = Time.from_msg(msg.orig_data_stamp)

        cones = conv.msg_to_cones(msg)
        self.merger.add(cones, PipelineType.ZED2_PIPELINE)

        return

    def lidar_cone_callback(self, msg):
        '''receive cones from lidar_node predictor'''
        self.data_times[PipelineType.LIDAR] = Time.from_msg(msg.orig_data_stamp)

        cones = conv.msg_to_cones(msg)
        self.merger.add(cones, PipelineType.LIDAR)

        return
    
    def flush_and_get_data_times(self):
        times = [self.data_times[datatype] for datatype in self.data_times.keys()]
        min_time = min(times, key=lambda t: t.nanoseconds)
        self.data_times = {}

        return min_time

    def publish_cones(self):

        # check cone publication conditions met
        if not self.merger.sufficient():
            self.get_logger().warn(f"Not got sufficient cones")
            return 

        # get the merged cones and reset 
        merged_cones = self.merger.merge()
        self.merger.reset()

        # update visualizer
        if self.debug:
            self.vis2D.set_cones(merged_cones)

        # publish cones
        msg = conv.cones_to_msg(merged_cones)

        data_time = self.flush_and_get_data_times()
        msg.orig_data_stamp = data_time.to_msg()

        self.cone_publisher.publish(msg)

        data_t = conv.ms_since_time(self.get_clock().now(), data_time)
        print(f"Published {len(merged_cones)} cones (Data Latency: {data_t}ms)")
        
        return
    
def start_cone_node(merger, args=None, debug=False):
    rclpy.init(args=args)

    cone_node = ConeNode(merger, debug=debug)

    rclpy.spin(cone_node)

    cone_node.destroy_node()
    rclpy.shutdown()

    return

def main_lidar(args=None):
    start_cone_node(create_lidar_merger(), args=args, debug=False)
    return

def main_lidar_debug(args=None):
    start_cone_node(create_lidar_merger(), args=args, debug=True)
    return

def main_zed(args=None):
    start_cone_node(create_zed_merger(), args=args, debug=False)
    return

def main_zed_debug(args=None):
    start_cone_node(create_zed_merger(), args=args, debug=True)
    return

def main_all(args=None):
    start_cone_node(create_all_merger(), args=args, debug=False)
    return

def main_all_debug(args=None):
    start_cone_node(create_all_merger(), args=args, debug=True)
    return

def main_any(args=None):
    start_cone_node(create_any_merger(), args=args, debug=False)
    return

def main_any_debug(args=None):
    start_cone_node(create_any_merger(), args=args, debug=True)
    return

# TODO: decide which policy is best and call it from main() (default to zed)
def main(args=None):
    main_zed()

if __name__ == "__main__":
    main()
