import rclpy
from rclpy.node import Node
from interfaces.msg import ConeArray, SplineFrames
import message_filters
import socket
import string
import os

class WebsocketNode(Node):
    def __init__(self, clientAddress, clientSocket):
        super().__init__('websocketer_node')
        self.cone_topic = "/perc_cones"
        self.midline_topic = "/spline"

        self.cone_subscribe = None
        self.midline_subscribe = None

        self.csocket = clientSocket
        print("Connected to", clientAddress)
        self.subscribe_topics()
        
    def subscribe_topics(self):
        self.cone_subscribe = message_filters.Subscriber(self, ConeArray, self.cone_topic)
        self.midline_subscribe = message_filters.Subscriber(self, SplineFrames, self.midline_topic)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.cone_subscribe, self.midline_subscribe], 10, 0.1)
        self.ts.registerCallback(self.midline_upload_callback)
        self.ts.registerCallback(self.cone_upload_callback)

    def midline_upload_callback(self, midline_msg):
        try:
            self.csocket.send(midline_msg)
            callback_confirmation = f"Message at {midline_msg.orig_data_stamp} Uploaded!"
        except socket.error:
            print("Error!")
            return 

    def cone_susbcription_callback(self, cone_msg):
        try:
            self.csocket.send(cone_msg)
            callback_confirmation = f"Message at {cone_msg.orig_data_stamp} Uploaded!"
        except socket.error:
            print("Error!")
            return 


def main(args=None):
    LOCALHOST = ''
    PORT = 9090
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((LOCALHOST, PORT))
    rclpy.init(args=args)

    while True:
        clientAddress, clientSocket = server.accept()
        websocket_node = WebsocketNode(clientAddress, clientSocket)
        rclpy.spin(websocket_node)
        websocket_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
