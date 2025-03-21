import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import json
import websockets
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import asyncio

NODE_NAME = 'pid_node'

BEST_EFFORT_QOS_PROFILE = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.VOLATILE,
    depth=5
)

class PIDTuning(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.ws_url = 'ws://live.cmr.red:2022'
        self.websocket = None
        self.connected = False
        
        self.pid_publisher = self.create_publisher(
            Point,
            '/pid_values',
            qos_profile=BEST_EFFORT_QOS_PROFILE
        )
        
        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.connect_websocket()

    def connect_websocket(self):
        try:
            self.websocket = self.loop.run_until_complete(
                websockets.connect(self.ws_url)
            )
            self.connected = True
            self.get_logger().info('Connected to WebSocket server')
        except Exception as e:
            self.get_logger().error(f'WebSocket connection failed: {str(e)}')
            self.connected = False

    def timer_callback(self):
        if not self.connected:
            self.connect_websocket()
            return

        try:
            message = self.loop.run_until_complete(self.websocket.recv())
            self.get_logger().info(f'yipeeee: {message}')

            data = json.loads(message)

            p = float(data['P'])
            i = float(data['I'])
            d = float(data['D'])
            
            point_msg = Point()
            point_msg.x = p
            point_msg.y = i
            point_msg.z = d
            
            self.pid_publisher.publish(point_msg)
            self.get_logger().info(f'Published PID values: P:{p} I:{i} D:{d}')
            
        except websockets.exceptions.ConnectionClosed:
            self.get_logger().warn('WebSocket connection closed')
            self.connected = False
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Invalid JSON message received: {str(e)}')
        except Exception as e:
            self.get_logger().error(f'Error: {str(e)}')
            self.connected = False

def main(args=None):
    rclpy.init(args=args)
    node = PIDTuning()
    
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f'Node error: {str(e)}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()