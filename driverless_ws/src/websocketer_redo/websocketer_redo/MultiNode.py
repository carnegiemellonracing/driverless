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
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import asyncio

BEST_EFFORT_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)

def main(args=None):
    HOST = 'live.cmr.red'
    PATH = '/22a'
    try: 
        websocket = connect("ws://live.cmr.red:2022")
        print("Connection with webserver established!")

        rclpy.init(args=args)

        websocket_node = rclpy.create_node('websocketer_node')
        cone_topic = "/perc_cones"
        midline_topic = "/spline"

        def cone_upload_callback(cone_msg):
            try:
                data = rosidl_runtime_py.convert.message_to_yaml(cone_msg)
                websocket.send("CONES START\n" + data + "CONES END\n")
                callback_confirmation = f"Message at {cone_msg.orig_data_stamp} Uploaded!"
            except socket.error:
                print("Error!")
                return

        def midline_upload_callback(midline_msg):
            try:
                data = rosidl_runtime_py.convert.message_to_yaml(midline_msg)
                websocket.send("MIDLINE START\n" + data + "MIDLINE END\n")
                callback_confirmation = f"Message at {midline_msg.orig_data_stamp} Uploaded!"
            except socket.error:
                print("Error!")
                return

        websocket_node.create_subscription(ConeArray, cone_topic, cone_upload_callback, BEST_EFFORT_QOS_PROFILE)
        websocket_node.create_subscription(SplineFrames, midline_topic, midline_upload_callback, BEST_EFFORT_QOS_PROFILE)

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
