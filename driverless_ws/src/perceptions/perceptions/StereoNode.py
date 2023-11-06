# ROS2 imports
import rclpy

# for subscribing to sensor data
from perceptions.PredictNode import PredictNode

# for converting predictor output to cone message type
from eufs_msgs.msg import ConeArray
import perceptions.conversions as conversions

# for doing prediction on sensor data
from perc22a.predictors.stereo.StereoPredictor import StereoPredictor

NODE_NAME = "stereo_node"

class StereoNode(PredictNode):

    def __init__(self):
        super().__init__(name=NODE_NAME)
        return

    def init_predictor(self):
        # create predictor
        self.model_name = 'ultralytics/yolov5'
        self.param_path = '/home/dale/driverless-packages/PerceptionsLibrary22a/perc22a/predictors/stereo/model_params.pt'
        predictor = StereoPredictor(self.model_name, self.param_path)
        return predictor

def main(args=None):
    rclpy.init(args=args)

    stereo_node = StereoNode()

    rclpy.spin(stereo_node)

    stereo_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()