# ROS2 imports
import rclpy

# for subscribing to sensor data
from perceptions.ros.utils.PredictNode import PredictNode

# for doing prediction on sensor data
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor

NODE_NAME = "lidar_node"

class LidarNode(PredictNode):

    def __init__(self, debug=False):
        super().__init__(name=NODE_NAME,  debug_flag=debug, time_flag=True)
        
        return
    
    def init_predictor(self):
        return LidarPredictor()

def main_debug(args=None):
    rclpy.init(args=args)

    stereo_node = LidarNode(debug=True)

    rclpy.spin(stereo_node)

    stereo_node.destroy_node()
    rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    stereo_node = LidarNode()

    rclpy.spin(stereo_node)

    stereo_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()