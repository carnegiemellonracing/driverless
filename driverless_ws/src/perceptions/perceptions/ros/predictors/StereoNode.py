# ROS2 imports
import rclpy

# for subscribing to sensor data
from perceptions.ros.utils.PredictNode import PredictNode

# for doing prediction on sensor data
from perc22a.predictors.stereo.StereoPredictor import StereoPredictor

NODE_NAME = "stereo_node"

class StereoNode(PredictNode):

    def __init__(self):
<<<<<<< HEAD
        super().__init__(name=NODE_NAME, debug_flag=False, time_flag=True)
=======
        super().__init__(name=NODE_NAME, debug_flag=True, time_flag=True)
>>>>>>> main
        return

    def init_predictor(self):
        # create predictor
        self.model_name = 'ultralytics/yolov5'
<<<<<<< HEAD
        self.param_path = '/home/chip/Desktop/Documents/driverless-packages/PerceptionsLibrary22a/perc22a/predictors/stereo/model_params.pt'
=======
        self.param_path = '/home/dale/driverless-packages/PerceptionsLibrary22a/perc22a/predictors/stereo/model_params.pt'
>>>>>>> main
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