import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from interfaces.msg import ControlAction

BEST_EFFORT_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 1)

class SteeringTest(Node):

    def __init__(self):
        super().__init__("steering_test_node")
        self.pub = self.create_publisher(msg_type=ControlAction,
                                         topic="/control_action",
                                         qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.time = 0.0
    
    def timer_callback(self):

        swangle = 0.3 * np.sin(2* np.pi * self.time / 4)
        self.time += 0.1
        pub_msg = ControlAction()
        pub_msg.swangle = swangle
        pub_msg.torque_fl = 2.0
        pub_msg.torque_fr = 2.0
        pub_msg.torque_rl = 0.0
        pub_msg.torque_rr = 0.0
        self.pub.publish(pub_msg)


def main(args=None):
    rclpy.init(args=args)

    test_node = SteeringTest()

    rclpy.spin(test_node)

    r = rclpy.Rate()

    test_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()