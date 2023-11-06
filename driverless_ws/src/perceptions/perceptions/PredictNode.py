# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

# for converting predictor output to cone message type
from eufs_msgs.msg import ConeArray
import perceptions.conversions as conversions

# for collecting data from sensors
from perceptions.DataNode import DataNode

import time

# configure QOS profile
BEST_EFFORT_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)

DEBUG = False
TIME = True

class PredictNode(DataNode):

    def __init__(self, name):
        super().__init__(name=name)

        self.name = name

        # TODO: figure out best way to time prediction appropriately
        self.interval = 0.1
        self.predict_timer = self.create_timer(self.interval, self.predict_callback)

        # initialize published cone topic based on name
        self.cone_topic = f"/{name}_cones"
        self.qos_profile = BEST_EFFORT_QOS_PROFILE
        self.cone_publisher = self.create_publisher(ConeArray, self.cone_topic, self.qos_profile)
        
        # create predictor, any subclass of PredictNode needs to fill this component
        self.predictor = self.init_predictor()

        return
    
    def init_predictor(self):
        raise RuntimeError("[PredictNode] init_predictor() function not overwritten. Must return Predictor.")

    def predict_callback(self):
        if not self.got_all_data():
            self.get_logger().warn(f"[Node={self.name}] Not got all data")
            return

        # predict cones from data
        s = time.time()
        cones = self.predictor.predict(self.data)
        e = time.time()

        # display if necessary
        if DEBUG:
            self.predictor.display()
            print(cones)

        # publish message
        msg = conversions.cones_to_msg(cones)
        self.cone_publisher.publish(msg)

        if TIME:
            # display time taken to perform prediction
            t = (e - s)
            time_str = f"[Node={self.name}] Predict Time: {t * 1000:.3f}ms"
            print(time_str)
        pass