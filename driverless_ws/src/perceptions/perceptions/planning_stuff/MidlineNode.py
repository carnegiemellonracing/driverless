
import rclpy
from rclpy.node import Node
from eufs_msgs.msg import ConeArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import perceptions.planning_stuff.svm_utils as svm_utils
from interfaces.msg import SplineFrames
from geometry_msgs import Point

import numpy as np


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
                                                 qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.midline_pub = self.create_publisher(msg_type=SplineFrames,
                                                 topic="/midline",
                                                 qos_profile=RELIABLE_QOS_PROFILE)
    
    def cone_callback(self, cones):
        blue = np.array()
        for cone in cones.blue_cones:
            blue.append([cone.x, cone.y, 0])

        yellow = []
        for cone in cones.yellow_cones:
            yellow.append([cone.x, cone.y, 1])
        
        data = np.vstack([np.array(blue), np.array(yellow)])
        print(data)

        downsampled_boundary_points = svm_utils.process(data)
        # print(downsampled_boundary_points)

        points = []
        msg = SplineFrames()

        for np_point in downsampled_boundary_points:
            new_point = Point(np_point[0], np_point[1], 0)
            points.append(new_point)

        msg.midpoints = points
        
        #TODO: add car pos to each midpoint to get global point

        self.midline_pub.publish(msg)


def main():
    print("hello world")
    rclpy.init()
    midline_node = MidlineNode()
    rclpy.spin(midline_node)

    midline_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()