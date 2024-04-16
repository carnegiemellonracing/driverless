import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from interfaces.msg import ControlAction
import numpy as np
import threading
import time


CMDLINE_QOS_PROFILE = QoSProfile(
    depth=1,  # Set the queue depth
    reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,  # Set the reliability policy
    durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE  # Set the durability policy
)

ACTION_DELAY = 5  # sec
BRAKE_THROTTLE = -20  # total
START_DELAY = 5

class SteeringTuningNode(Node):

    def __init__(self):
        super().__init__('steering_tuning')

        self.publisher = self.create_publisher(
            ControlAction,
            '/control_action',
            CMDLINE_QOS_PROFILE
        )

        self.swangles = [0, np.radians(15), np.radians(-15)]
        self.end_action = (0., BRAKE_THROTTLE / 2., BRAKE_THROTTLE / 2., 0., 0.)

    def publish(self, action):
        msg = ControlAction()
        print(f"Publishing {action}")
        msg.swangle = action[0]
        msg.torque_fl = action[1]
        msg.torque_fr = action[2]
        msg.torque_rl = action[3]
        msg.torque_rr = action[4]
        self.publisher.publish(msg)

    def run(self, throttle):
        time.sleep(START_DELAY)
        
        for swangle in self.swangles:
            self.publish(np.array([swangle, throttle / 2, throttle / 2, 0.0, 0.0]))
            time.sleep(ACTION_DELAY)

        self.publish(self.end_action)
        time.sleep(ACTION_DELAY)


def main(args=None):
    rclpy.init(args=args)

    publisher = SteeringTuningNode()
    spinner = threading.Thread(target=rclpy.spin, args=(publisher,))
    spinner.start()

    try:
        throttle = int(input("Enter throttle (total, Nm): ").strip())
        publisher.run(throttle)

    except BaseException as e:
        print(e)

    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
