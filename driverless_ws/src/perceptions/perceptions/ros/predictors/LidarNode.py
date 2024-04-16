# ROS2 imports
import rclpy

# for subscribing to sensor data
from perceptions.ros.utils.PredictNode import PredictNode

# for doing prediction on sensor data
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor
from perc22a.predictors.lidar.FMSLidarPredictor import FMSLidarPredictor

NODE_NAME = "lidar_node"

class LidarNode(PredictNode):

    def __init__(self, debug=False, fms=False):
        self.fms = fms

        super().__init__(name=NODE_NAME, debug_flag=debug, time_flag=True)
        return
    
    def init_predictor(self):
        return FMSLidarPredictor() if self.fms else LidarPredictor()

def start_node(args, debug=False, fms=False):
    rclpy.init(args=args)

    lidar_node = LidarNode(debug=debug, fms=fms)

    rclpy.spin(lidar_node)

    lidar_node.destroy_node()
    rclpy.shutdown()

def main(args=None):
    start_node(args, debug=False, fms=False)

def main_debug(args=None):
    start_node(args, debug=True, fms=False)

def main_fms(args=None):
    start_node(args, debug=False, fms=True)

def main_fms_debug(args=None):
    start_node(args, debug=True, fms=True)


if __name__ == "__main__":
    main()