import rclpy
from rclpy.node import Node
import torch
import numpy as np
import time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from stereo.predict import predict
# from stereo.ZED import ZEDSDK

from interfaces.msg import ConeArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image

from cv_bridge import CvBridge

def np2points(cones):
    '''convert list of cones into a Point[] object'''
    arr = []
    for i in range(len(cones)):
        p = Point()
        p.x = float(cones[i][0])
        p.y = float(cones[i][1])
        p.z = float(cones[i][2])

        arr.append(p)
    return arr

BEST_EFFORT_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)
STEREO_OUT = '/stereo_cones'
IMAGE_LEFT_OUT = '/zedsdk_left_color_image'
IMAGE_RIGHT_OUT = '/zedsdk_right_color_image'
DEPTH_OUT = '/zedsdk_depth_image'
POINT_OUT = '/zedsdk_point_cloud_image'

class StereoCamera(Node):

    def __init__(self):
        super().__init__('stereo_predictor')
        frame_rate = 20
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='src/stereo/stereo/model_params.pt')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        self.prediction_publisher = self.create_publisher(msg_type=ConeArray,
                                                          topic=STEREO_OUT,
                                                          qos_profile=BEST_EFFORT_QOS_PROFILE)
        # self.left_publisher = self.create_publisher(msg_type=Image,
        #                                              topic=IMAGE_LEFT_OUT,
        #                                              qos_profile=BEST_EFFORT_QOS_PROFILE)
        # self.right_publisher = self.create_publisher(msg_type=Image,
        #                                              topic=IMAGE_RIGHT_OUT,
        #                                              qos_profile=BEST_EFFORT_QOS_PROFILE)
        # self.depth_publisher = self.create_publisher(msg_type=Image,
        #                                              topic=DEPTH_OUT,
        #                                              qos_profile=BEST_EFFORT_QOS_PROFILE)
        # self.point_publisher = self.create_publisher(msg_type=Image,
        #                                              topic=POINT_OUT,
        #                                              qos_profile=BEST_EFFORT_QOS_PROFILE)
    
        self.left_sub = self.create_subscription(msg_type=Image, topic=IMAGE_LEFT_OUT, callback=self.left_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.point_sub = self.create_subscription(msg_type=Image, topic=POINT_OUT, callback=self.point_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)

        

        self.data_syncer = self.create_timer(1/frame_rate, self.inference)

        # self.zed = ZEDSDK()
        # self.zed.open()

        self.bridge = CvBridge()

        self.left_msg = None
        self.point_msg = None


        print(f"model-device: {self.device}")
        print("done-init-node")

    def left_callback(self, msg):
        self.left_msg = msg

    def point_callback(self, msg):
        self.point_msg = msg

    def inference(self):
        # try displaying the image

        s = time.time()

        # left, right, depth, point = self.zed.grab_data()
        # blue_cones, yellow_cones, orange_cones = predict(self.model, left, point)

        # convert the data and check that it is the same going and backwards
        # have to extract out nan values that don't count to compare image values
        if (self.left_msg is None or self.point_msg is None):
            return
        
        left_enc = self.left_msg
        point_enc = self.point_msg

        # publish the data
        # self.left_publisher.publish(left_enc)
        # self.right_publisher.publish(right_enc)
        # self.depth_publisher.publish(depth_enc)
        # self.point_publisher.publish(point_enc)

        left_unenc = self.bridge.imgmsg_to_cv2(left_enc, desired_encoding="passthrough")
        point_unenc = self.bridge.imgmsg_to_cv2(point_enc, desired_encoding="32FC4")

        blue_cones, yellow_cones, orange_cones = predict(self.model, left_unenc, point_unenc)

        print(blue_cones, yellow_cones, orange_cones)

        result = []
        for i in range(len(blue_cones)):
            c = blue_cones[i]
            result.append([c[0], c[1], c[2]])
        print(np.array(result))

        cone_msg = ConeArray()
        cone_msg.blue_cones = np2points(blue_cones)
        cone_msg.yellow_cones = np2points(yellow_cones)
        cone_msg.orange_cones = np2points(orange_cones)
        self.prediction_publisher.publish(cone_msg)

        t = time.time()
        print(f"Stereo: {1000 * (t - s):.3f}ms")

        

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
