import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from interfaces.msg import ControlAction

class MyPublisher(Node):

    def __init__(self):
        super().__init__('my_publisher')
        self.publisher_ = self.create_publisher(ControlAction, '/control_action', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.direction = 1

    def timer_callback(self):
        msg = ControlAction()
        msg.torque_fl = 1000.0
        msg.torque_fr = 1000.0
        msg.torque_rl = 1000.0
        msg.torque_rr = 1000.0

        if (self.i >= 20 or self.i <= -20):
            self.direction = -self.direction
        
        msg.swangle = float(self.i + self.direction)*3.141592/180
        self.i = self.i + self.direction
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.swangle}')

def main(args=None):
    rclpy.init(args=args)
    my_publisher = MyPublisher()
    rclpy.spin(my_publisher)
    my_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
