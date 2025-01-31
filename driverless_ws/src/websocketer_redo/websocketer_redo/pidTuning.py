#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class PIDInputPublisher(Node):
    def __init__(self):
        super().__init__('pid_input_publisher')
        
        # Configure QoS profile for better reliability
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create publisher for PID constants
        self.pid_publisher = self.create_publisher(
            Float32MultiArray,
            'pid_constants',
            qos_profile
        )
        
        self.get_logger().info('PID Input Publisher Node started')
        self.get_logger().info('Enter normalized PID values (0-1) when prompted')
    
    def get_valid_input(self, prompt):
        while True:
            try:
                value = float(input(prompt))
                if 0 <= value <= 1:
                    return value
                else:
                    print("Error: Value must be between 0 and 1")
            except ValueError:
                print("Error: Please enter a valid number")
    
    def publish_pid_constants(self):
        # Get PID values from user
        p = self.get_valid_input("Enter normalized P value (0-1): ")
        i = self.get_valid_input("Enter normalized I value (0-1): ")
        d = self.get_valid_input("Enter normalized D value (0-1): ")
        
        # Create message
        msg = Float32MultiArray()
        msg.data = [p, i, d]
        
        # Publish message
        self.pid_publisher.publish(msg)
        self.get_logger().info(f'Published PID constants: P={p}, I={i}, D={d}')

def main(args=None):
    rclpy.init(args=args)
    node = PIDInputPublisher()
    
    try:
        while rclpy.ok():
            node.publish_pid_constants()
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()