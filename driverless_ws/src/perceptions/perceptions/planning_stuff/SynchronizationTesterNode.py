
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from interfaces.msg import ConeArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import perceptions.planning_stuff.svm_utils as svm_utils
from interfaces.msg import SplineFrames
from geometry_msgs.msg import QuaternionStamped
from sensor_msgs.msg import PointCloud2

import perceptions.ros.utils.conversions as conv
from perc22a.svm.svm_utils import cones_to_midline
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



class SynchronizationTesterNode(Node):

    def __init__(self):
        super().__init__("midline_node")
        self.quat_sub = self.create_subscription(msg_type=QuaternionStamped,
                                                 topic="/filter/quaternion",
                                                 callback=self.quat_callback,
                                                 qos_profile=RELIABLE_QOS_PROFILE)
        
        self.lidar_sub = self.create_subscription(msg_type=PointCloud2,
                                                 topic="/lidar_points",
                                                 callback=self.lidar_callback,
                                                 qos_profile=RELIABLE_QOS_PROFILE)

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.curr_diff = 0
        self.max_ddiff = 0
        self.d_diffs = []

    def quat_callback(self, msg):
        # print("Recieiving cone callback in midline")

        self.recent_quat = msg
    
    def lidar_callback(self, msg):
        # print("Recieiving cone callback in midline")

        self.recent_lidar = msg

    def timer_callback(self):
        quat_timestamp = self.recent_quat.header.stamp.sec + self.recent_quat.header.stamp.nanosec * 1e-9 
        lidar_timestamp = self.recent_lidar.header.stamp.sec + self.recent_lidar.header.stamp.nanosec * 1e-9
        self.prev_diff = self.curr_diff
        self.curr_diff = quat_timestamp - lidar_timestamp
        self.d_diff = abs(self.curr_diff - self.prev_diff)
        self.d_diffs.append(self.d_diff)
        print(f"quat: {quat_timestamp:.3f}")
        print(f"lidar: {lidar_timestamp:.3f}")
        print(f"diff: {self.curr_diff:.3f}")
        print(f"d_diff: {self.d_diff:.3f}")
        if len(self.d_diffs) > 1:
            print(f"max d_diff: {max(self.d_diffs[1:]):.3f}")
            print(f"avg d_diff: {sum(self.d_diffs[1:]) / len(self.d_diffs):.3f}")
            print(f"n: {len(self.d_diffs)}")
        print("--------------------------------------------")

def main():
    rclpy.init()
    midline_node = SynchronizationTesterNode()
    rclpy.spin(midline_node)

    midline_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()