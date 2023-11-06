# ROS2 imports
import rclpy

# for subscribing to sensor data
from perceptions.PredictNode import PredictNode

# for converting predictor output to cone message type
from eufs_msgs.msg import ConeArray
import perceptions.conversions as conversions

# for doing prediction on sensor data
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor

NODE_NAME = "lidar_node"

class LidarNode(PredictNode):

    def __init__(self):
        super().__init__(name=NODE_NAME)
        
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