import throttleUSB
import rclpy
from interfaces.msg import ControlAction
import numpy as np


class SwangleInputNode(rclpy.Node):
    def __init(self):
        super().__init__('swangle_input')

        self.publisher = self.create_publisher(
            ControlAction,
            '/control_action',  # Replace with the desired topic name
            throttleUSB.CMDLINE_QOS_PROFILE
        )

    def publish(self, swangle):
        action = ControlAction()
        action.swangle = np.radians(swangle)
        action.torque_fl = -100
        action.torque_fr = -100
        action.torque_rl = -100
        action.torque_rr = -100
        self.publisher.publish(action)


def main():
    publisher = SwangleInputNode()
    rclpy.spin(publisher)

    try:
        while True:
            swangle = input("Swangle (deg): ")
            publisher.publish(swangle)

    except BaseException:
        pass

    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
