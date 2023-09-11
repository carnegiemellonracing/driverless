import rclpy
from rclpy.node import Node
import torch
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from driverless_ws.src.eufs_msgs.msg import ConeList

from src.stereo.stereo.predict import predict

from ZED import ZEDSDK

BEST_EFFORT_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)
STEREO_OUT = '/stereo_cones'

class StereoCamera(Node):

    def __init__(self):
        super().__init__('stereo_predictor')
        frame_rate = 20
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_april_27.pt')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        self.prediction_publisher = self.create_publisher(msg_type=ConeList,
                                                          topic=STEREO_OUT,
                                                          qos_profile=BEST_EFFORT_QOS_PROFILE)

        self.data_syncer = self.create_timer(1/frame_rate, self.inference)

        self.zed = ZEDSDK()


        print(f"sim: {self.sim} model-device: {self.device}")
        print("done-init-node")

    def inference(self):
        # try displaying the image

        blue_cones, yellow_cones, orange_cones = predict(self.model, self.zed)
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
