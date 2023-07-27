from xmlrpc.server import DocXMLRPCRequestHandler
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Int8
from nav_msgs.msg import Odometry #TODO:make sure that the driver is actually installed
# from sbg_driver.msg import SbgGpsPos #TODO: make sure that the sbg drivers properly installed
import message_filters #TODO Make sure that this is installed too
from cmrdv_interfaces.msg import VehicleState, ConePositions,ConeList #be more specific later if this becomes huge
from cmrdv_common.config.collection_config import BEST_EFFORT_QOS_PROFILE
from cmrdv_common.config.planning_config import *
# from cmrdv_ws.src.cmrdv_planning.planning_codebase.ekf.map import *
from cmrdv_ws.src.cmrdv_planning.planning_codebase.ekf.new_slam import *
from cmrdv_ws.src.cmrdv_planning.planning_codebase.graph_slam import *
from cmrdv_interfaces.msg import *
import numpy as np
import math
import time
from transforms3d.quaternions import axangle2quat


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription_cone_data = message_filters.Subscriber(self, ConeArrayWithCovariance, '/cones')
        self.subscription_vehicle_data = message_filters.Subscriber(self, CarState, '/ground_truth/state')
        self.ts = message_filters.ApproximateTimeSynchronizer([self.subscription_cone_data, self.subscription_vehicle_data], 10, slop=0.05)
        self.ts.registerCallback(self.visualize)
    def visualize(self, cones_msg, car_state_msg):
        pass
    
    def calc_landmark_position(x, z):
        # x = [x, y, theta] for car
        # z = 
        zp = np.zeros((2, 1))

        zp[0, 0] = x[0, 0] + z[0] * math.cos(x[2, 0] + z[1])
        zp[1, 0] = x[1, 0] + z[0] * math.sin(x[2, 0] + z[1])

        return zp

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()