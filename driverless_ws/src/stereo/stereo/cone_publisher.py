import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from eufs_msgs.msg import ConeArrayWithCovariance, ConeWithCovariance
import math 

class ConePublisher(Node):

    def __init__(self):
        super().__init__('conePublisher')
        self.publisher_ = self.create_publisher(ConeArrayWithCovariance, 'cones', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = ConeArrayWithCovariance()

        x0 = self.i
        y0 = (x0/10) * math.sin(x0/10)

        msg.blue_cones = [
            ConeWithCovariance(
                point=Point(x=float(x0), y=float(0), z=float(y0 + 3)), 
                covariance=[0.0, 0.0, 0.0, 0.0])
        ]

        msg.yellow_cones = [
            ConeWithCovariance(
                point=Point(x=float(x0), y=float(0), z=float(y0 - 3)), 
                covariance=[0.0, 0.0, 0.0, 0.0])
        ]
        self.publisher_.publish(msg)
        self.i += 3

def main(args=None):
    rclpy.init(args=args)

    conePublisher = ConePublisher()

    rclpy.spin(conePublisher)
    conePublisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()