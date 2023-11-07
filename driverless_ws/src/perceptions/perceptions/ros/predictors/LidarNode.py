# ROS2 imports
import rclpy

# for subscribing to sensor data
from perceptions.ros.utils.PredictNode import PredictNode

# for doing prediction on sensor data
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor

NODE_NAME = "lidar_node"

class LidarNode(PredictNode):

    def __init__(self):
        super().__init__(name=NODE_NAME,  debug_flag=False, time_flag=True)
        
        return
    
    def init_predictor(self):
        return LidarPredictor()


def main(args=None):
    rclpy.init(args=args)

    stereo_node = LidarNode()

    rclpy.spin(stereo_node)

    stereo_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()