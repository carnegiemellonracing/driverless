import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
import numpy as np
import numpy_ros as np_ros

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Point, 'numpy-test', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        arr = np.asarray([1.0, 2.0, 3.0])
        msg = np_ros.to_message(Point, arr)
        self.publisher_.publish()
        print(f'publishing: {msg.x}, {msg.y}, {msg.z}')

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
