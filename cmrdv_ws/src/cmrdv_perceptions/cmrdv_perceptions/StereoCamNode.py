import rclpy
from rclpy.node import Node
import torch
from std_msgs.msg import String
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config import collection_config as collection_cfg
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config import perceptions_config as perceptions_cfg
from cmrdv_ws.src.cmrdv_perceptions.stereo_vision.predict import predict
from cmrdv_ws.src.cmrdv_perceptions.stereo_vision.ZED import ZEDSDK
from cmrdv_ws.src.cmrdv_perceptions.utils.utils import np2points
from cmrdv_interfaces.msg import DataFrame, SimDataFrame, ConeList
from cmrdv_ws.src.cmrdv_common.cmrdv_common.DIM.dim_heartbeat import HeartbeatNode

import time
import cv2
from cmrdv_ws.src.cmrdv_common.cmrdv_common.conversions import image_to_numpy
import numpy as np

class StereoCamera(HeartbeatNode):

    def __init__(self):
        super().__init__('stereo_predictor')
        self.declare_parameter('use_simulated_data', False)

        frame_rate = 20

        self.sim = self.get_parameter('use_simulated_data').get_parameter_value().bool_value
        self.zed = ZEDSDK()
        self.zed.open()
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=perceptions_cfg.YOLO_WEIGHT_FILE)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        self.prediction_publisher = self.create_publisher(msg_type=ConeList,
                                                          topic=perceptions_cfg.STEREO_OUT,
                                                          qos_profile=collection_cfg.BEST_EFFORT_QOS_PROFILE)

        self.data_syncer = self.create_timer(1/frame_rate, self.inference)

        print(f"sim: {self.sim} model-device: {self.device}")
        print("done-init-node")

    def inference(self):
        # try displaying the image

        blue_cones, yellow_cones, orange_cones = predict(self.model, self.zed)

        s = f"#blue: {len(blue_cones)}, #yellow: {len(yellow_cones)}, #orange: {len(orange_cones)}"
        self.get_logger().info(s)

        result = []
        for i in range(len(blue_cones)):
            c = blue_cones[i]
            result.append([c[0], c[1], c[2]])
        print(np.array(result))

        cone_msg = ConeList()
        cone_msg.blue_cones = np2points(blue_cones)
        cone_msg.yellow_cones = np2points(yellow_cones)
        cone_msg.orange_cones = np2points(orange_cones)
        self.prediction_publisher.publish(cone_msg)
        

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = StereoCamera()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
