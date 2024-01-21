# ROS2 imports
import rclpy

# for subscribing to sensor data
from perceptions.ros.utils.PredictNode import PredictNode

# for doing prediction on sensor data
from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor

NODE_NAME = "yolov5_node"

class StereoNode(PredictNode):

    def __init__(self):
        super().__init__(name=NODE_NAME, debug_flag=True, time_flag=True)
        return

    def init_predictor(self):
        return YOLOv5Predictor()

def main(args=None):
    rclpy.init(args=args)

    stereo_node = StereoNode()

    rclpy.spin(stereo_node)

    stereo_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()