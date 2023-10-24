# ROS2 imports
import rclpy

# for subscribing to sensor data
from perceptions.utils.DataNode import DataNode

# for doing prediction on sensor data
from perc22a.predictors import LidarPredictor

class LidarNode(DataNode):

    def __init__(self):
        super().__init__(name="stereo_node")

        # do prediction on a timer
        self.interval = 1
        self.predict_timer = self.create_timer(self.interval, self.predict_callback)

        # create predictor
        self.predictor = LidarPredictor()

    def predict_callback(self):
        if not self.got_all_data():
            self.get_logger().warn("Not got all data")
            return
    
        # otherwise, do prediction on data
        data = {
            "points": self.points
        }
        cones = self.predictor.predict(data)

        self.predictor.display()
        print(cones)


def main(args=None):
    rclpy.init(args=args)

    stereo_node = LidarNode()

    rclpy.spin(stereo_node)

    stereo_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()