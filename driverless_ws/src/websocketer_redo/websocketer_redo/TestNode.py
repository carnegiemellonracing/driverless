import rclpy
from rclpy.node import Node
from interfaces.msg import ConeArray, SplineFrames
import message_filters
import rosidl_runtime_py
import socket
import string
from websockets.sync.client import connect
import os
import json

import asyncio

class WebsocketNode(Node):
    def __init__(self):
        super().__init__('websocketer_node')
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.i = 0
        
    def timer_callback(self):
        print(f"Hello {self.i}")
        self.i += 1


def main(args=None):
    HOST = 'live.cmr.red'
    PATH = '/22a'
    try: 
        # websocket = connect("ws://live.cmr.red:2022")
        # print("Connection with webserver established!")

        rclpy.init(args=args)

        websocket_node = WebsocketNode()

        rclpy.spin(websocket_node)
        websocket_node.destroy_node()
        rclpy.shutdown()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()
    rclpy.init(args=args)


if __name__ == '__main__':
    main()
