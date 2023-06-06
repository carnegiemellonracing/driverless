import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from cmrdv_interfaces.msg import DataFrame, SimDataFrame
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.planning_config import * # Just CARROT_TOPIC for now
from cmrdv_interfaces.msg import CarROT, VehicleState
import numpy as np



#This Node might not be needed
#Both Publisher and Subscriber
class PathOut(Node):

    def __init__(self):
        super().__init__('path_out')
        self.subscription_midline = self.create_subscription(
            np.array, #TBD by perceptions
            MIDLINE_TOPIC,
            self.parseCarrotInput,
            queue_size = 10)
        self.subscription_midline  # prevent unused variable warning


        self.subscription_raceline = self.create_subscription(
            np.array, #TBD by perceptions
            RACELINE_PATH_TOPIC,
            self.parseCarrotInput,
            queue_size = 10)
        self.subscription_raceline  # prevent unused variable warning
        #subscribing to both slam and midline data

        #Carrot needs pose and curvature
        self.publisher_ = self.create_publisher(CARROT_TOPIC,CarROT, queue_size=10)
        #we publish whenever we get data, so we don't need to send on a periodic basis
    
    def parseCarrotInput(self,msg):
        #Nothing to parse for now
        #carrotData = msg.data
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing path to controls')
        #publishing the path planning data to controls



def main(args=None):
    rclpy.init(args=args)

    path_out = PathOut()
    rclpy.spin(path_out)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    path_out.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()