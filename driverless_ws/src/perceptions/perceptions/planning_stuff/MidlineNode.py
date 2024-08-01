import rclpy
from rclpy.node import Node
from rclpy.time import Time
from interfaces.msg import ConeArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import perceptions.planning_stuff.svm_utils as svm_utils
from interfaces.msg import SplineFrames
from geometry_msgs.msg import Point

import perceptions.ros.utils.conversions as conv
from perc22a.svm.svm_utils import cones_to_midline
from perc22a.svm.SVM import SVM
from perc22a.predictors.utils.vis.Vis2D import Vis2D

import numpy as np
import time


# configure QOS profile
BEST_EFFORT_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)

RELIABLE_QOS_PROFILE = QoSProfile(
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
)



class MidlineNode(Node):

    def __init__(self):
        super().__init__("midline_node")
        self.cone_sub = self.create_subscription(msg_type=ConeArray,
                                                 topic="/perc_cones",
                                                 callback=self.cone_callback,
                                                 qos_profile=RELIABLE_QOS_PROFILE)
        self.midline_pub = self.create_publisher(msg_type=SplineFrames,
                                                 topic="/spline",
                                                 qos_profile=RELIABLE_QOS_PROFILE)
        self.svm = SVM()
        # self.vis = Vis2D()
        self.failure_count = 0


    def cone_callback(self, msg):

        s = time.time()

        orig_data_stamp = msg.orig_data_stamp

        # parse cones from message received by cone_sub into cones variable
        cones = conv.msg_to_cones(msg)
        print("num cones", len(cones))


        # svm generates decision boundary (midline) from cones
        s_svm = time.time()
        downsampled_boundary_points = self.svm.cones_to_midline(cones)
        e_svm = time.time()


        # parse the elements in downsampled_boundary_points into Point types
        # points form midline_spline published by midline_pub
        points = []
        msg = SplineFrames()
        msg.orig_data_stamp = orig_data_stamp

        for np_point in downsampled_boundary_points:
            new_point = Point()
            new_point.x = float(np_point[0]) # turning into not SAE coordinates
            new_point.y = float(np_point[1])
            new_point.z = float(0)
            points.append(new_point)

        if len(points) < 2:
            print("LESS THAN 2 FRAMES")
            return

        msg.frames = points
        self.midline_pub.publish(msg)

        e = time.time()

        data_stamp = orig_data_stamp
        data_t = Time.from_msg(orig_data_stamp)
        time_since_data = conv.ms_since_time(self.get_clock().now(), data_t)
        print(f"Entire: {int(1000 * (e - s)):.3f}, Data Latency: {time_since_data:.3f}ms, # frames: {len(points)} Failures: {self.failure_count}")


def main():
    rclpy.init()
    midline_node = MidlineNode()
    rclpy.spin(midline_node)

    midline_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
