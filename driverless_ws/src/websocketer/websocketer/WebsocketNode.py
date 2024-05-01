import rclpy
from rclpy.node import Node
from interfaces.msg import ConeArray, SplineFrames
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import message_filters
import rosidl_runtime_py
import socket
import string
from websockets.sync.client import connect
import os
import json

import asyncio

BEST_EFFORT_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 1)

class WebsocketNode(Node):
    def __init__(self, socket):
        super().__init__('websocketer_node')
        self.cone_topic = "/perc_cones"
        self.midline_topic = "/spline"

        self.cone_subscribe = None
        self.midline_subscribe = None

        self.socket = socket
        print("Connected to", self.socket)
        self.subscribe_topics()
        
    def subscribe_topics(self):
        self.cone_subscribe = message_filters.Subscriber(self, ConeArray, self.cone_topic, qos_profile=BEST_EFFORT_QOS_PROFILE)
        # self.midline_subscribe = message_filters.Subscriber(self, SplineFrames, self.midline_topic)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.cone_subscribe], 10, 0.1)
        # self.ts.registerCallback(self.midline_upload_callback)
        self.ts.registerCallback(self.cone_upload_callback)

    def midline_upload_callback(self, midline_msg):
        try:
            self.socket.send(json.dumps(midline_msg))
            callback_confirmation = f"Message at {midline_msg.orig_data_stamp} Uploaded!"
        except socket.error:
            print("Error!")
            return 

    def cone_upload_callback(self, cone_msg):
        try:
            data = rosidl_runtime_py.convert.message_to_yaml(cone_msg)
            print(data)
            self.socket.send(data)
            callback_confirmation = f"Message at {cone_msg.orig_data_stamp} Uploaded!"
        except socket.error:
            print("Error!")
            return 


def main(args=None):
    HOST = 'live.cmr.red'
    PATH = '/22a'
    try: 
        websocket = connect("ws://live.cmr.red:2022")
        print("Connection with webserver established!")

        rclpy.init(args=args)

        websocket_node = WebsocketNode(websocket)

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