
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from interfaces.msg import TrackBounds, SplineFrames
from plan22a.global_racetrajectory_optimization.main_globaltraj import main_globaltraj
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

class RacelineNode(Node):
    def __init__(self):
        super().__init__("raceline node")
        self.track_bounds_sub = self.create_subscription(msg_type=TrackBounds,
                                                         topic="/track_bounds",
                                                         callback=self.track_bounds_callback,
                                                         qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.raceline_pub = self.create_publisher(msg_type=SplineFrames,
                                                  topic="/midline",
                                                  qos_profile=RELIABLE_QOS_PROFILE)
        
    def track_bounds_callback(self, track_bounds):
        tum_input = []
        for track_bound in track_bounds.bounds:
            left = track_bound.left
            right = track_bound.right
            dist = np.sqrt((left.x - right.x)**2 + (left.y - right.y)**2)
            midpoint_x = (left.x + right.x) / 2
            midpoint_y = (left.y + right.y) / 2
            tum_input.append([midpoint_x, midpoint_y, dist/2, dist/2])
        
        tum_input = np.array(tum_input)
        tum_output = main_globaltraj(tum_input)
        msg = SplineFrames()
        msg.fastmode = True
        raceline_points = []
        for raceline_point in tum_output:
            point = Point()
            point.x = float(raceline_point[0])
            point.y = float(raceline_point[1])
            raceline_points.append(point)
        msg.points = raceline_points
        self.raceline_pub.publish(msg)

def main():
    rclpy.init()
    raceline_node = RacelineNode()
    rclpy.spin(raceline_node)

    raceline_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()