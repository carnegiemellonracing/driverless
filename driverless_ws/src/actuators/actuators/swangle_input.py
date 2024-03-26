import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from interfaces.msg import ControlAction
import numpy as np
import threading


CMDLINE_QOS_PROFILE = QoSProfile(
    depth=1,  # Set the queue depth
    reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,  # Set the reliability policy
    durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE  # Set the durability policy
)

class SwangleInputNode(Node):
    def __init__(self):
        super().__init__('swangle_input')

        self.publisher = self.create_publisher(
            ControlAction,
            '/control_action',  # Replace with the desired topic name
            CMDLINE_QOS_PROFILE
        )

    def publish(self, swangle):
        swangle = float(swangle)
        action = ControlAction()
        action.swangle = np.radians(swangle)
        action.torque_fl = -100.0
        action.torque_fr = -100.0
        action.torque_rl = -100.0
        action.torque_rr = -100.0
        self.publisher.publish(action)

def main(args=None):
    rclpy.init(args=args)

    publisher = SwangleInputNode()
    spinner = threading.Thread(target=rclpy.spin, args=(publisher,))
    spinner.start()

    try:
        while True:
            swangle = input("Swangle (deg): ")
            publisher.publish(swangle)

    except BaseException as e:
        print(e)

    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
