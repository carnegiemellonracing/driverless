import rclpy
from rclpy.node import Node
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.collection_config import BEST_EFFORT_QOS_PROFILE
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.planning_config import PAIRROT_TOPIC
from cmrdv_interfaces.msg import PairROT


class StateSpoofer(Node):

    def __init__(self, state):
        super().__init__('state_spoofer')
        self.state = state

        self.pairrot_publisher = self.create_publisher(
            PairROT,
            PAIRROT_TOPIC, 
            BEST_EFFORT_QOS_PROFILE
        )

        pairrot_timer_period = 0.1  # seconds
        self.pairrot_timer = self.create_timer(pairrot_timer_period, self.pairrot_timer_callback)

    def pairrot_timer_callback(self):
        msg = PairROT()

        msg.near.x = 1.
        msg.near.y = 3.
        msg.near.yaw = 1.
        msg.near.curvature = 0.1

        self.pairrot_publisher.publish(msg)


def main(args=None):
    state = [0, 0, 0, 0, 0, 0]
    
    rclpy.init(args=args)

    minimal_publisher = StateSpoofer(state)

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
