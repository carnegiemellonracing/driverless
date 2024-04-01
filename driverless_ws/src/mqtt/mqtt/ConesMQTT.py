# code pulled and modified from https://github.com/robofoundry/ros2_mqtt/blob/main/ros2_mqtt/ros2_mqtt/relay_ros2_mqtt.py

import rclpy
from rclpy.time import Time

from time import sleep
import sys
import threading
import numpy as np
import os
print("hellow")
import paho.mqtt.client as mqtt
from rclpy.node import Node

from interfaces.msg import ConeArray
# import perceptsions.ros.utils.conversions as conversions
import json

BEST_EFFORT_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)

NODE_NAME = 'cones_mqtt'
SUBSCRIPTION_TOPIC = '/perc_cones'
PUBLISH_TOPIC = '/mqtt_topics/perc_cones'

class ConesMQTT(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        
        self.sleep_rate = 0.025
        self.rate = 10
        self.r = self.create_rate(self.rate)

        # set subscription and publish topics
        self.MQTT_PUB_TOPIC = self.declare_parameter("~mqtt_pub_topic", PUBLISH_TOPIC).value
        self.ROS_TWIST_SUB_TOPIC = self.declare_parameter("~ros2_sub_topic", SUBSCRIPTION_TOPIC).value
        self.mqttc = mqtt.Client(protocol=mqtt.MQTTv5)
        self.mqttc.tls_set(
            ca_certs=args.ca,
            certfile=args.certificate,
            keyfile=args.key,
            tls_version=2)
        self.mqttc.on_connect = on_connect
        self.mqttc.on_message = on_message
        self.mqttc.connect(args.endpoint, 8883, 60)

        self.create_subscription(
            ConeArray,
            self.ROS_TWIST_SUB_TOPIC,
            self.publish_to_mqtt,
            qos_profile=BEST_EFFORT_QOS_PROFILE)

        # instantiation callbacks
        self.get_logger().info('cones_mqtt:: started...')
        self.get_logger().info(f'cones_mqtt:: MQTT_PUB_TOPIC = {self.MQTT_PUB_TOPIC}')
        self.get_logger().info(f'cones_mqtt:: ROS_TWIST_SUB_TOPIC = {self.ROS_TWIST_SUB_TOPIC}')

    def publish_to_mqtt(self, tmsg):
        self.get_logger().info(tmsg)
        self.mqttclient.publish(self.MQTT_PUB_TOPIC, tmsg, qos=0, retain=False)

def main(args=None):

    rclpy.init(args=args)
    try:
        cones_mqtt = ConesMQTT()
        rclpy.spin(cones_mqtt)
    except rclpy.exceptions.ROSInterruptException:
        pass

    cones_mqtt.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()