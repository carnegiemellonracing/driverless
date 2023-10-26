# ROS2 imports
import rclpy

# for subscribing to sensor data
from perceptions.utils.DataNode import DataNode

# for converting predictor output to cone message type
from eufs_msgs.msg import ConeArrayWithCovariance
import perceptions.utils.conversions as conversions

# for doing prediction on sensor data
from perc22a.predictors import LidarPredictor

NODE_NAME = "lidar_node"

class LidarNode(DataNode):

    def __init__(self):
        super().__init__(name=NODE_NAME)

        # do prediction on a timer
        self.interval = 0.5
        self.predict_timer = self.create_timer(self.interval, self.predict_callback)

        # create publisher
        self.cone_topic = f"/{NODE_NAME}_cones"
        self.qos_profile = 10
        self.cone_publisher = self.create_publisher(ConeArrayWithCovariance, self.cone_topic, self.qos_profile)

        # create predictor
        self.predictor = LidarPredictor()

    def predict_callback(self):
        if not self.got_all_data():
            self.get_logger().warn("Not got all data")
            return
    
        # otherwise, do prediction on data and display
        data = {
            "points": self.points
        }
        cones = self.predictor.predict(data)
        self.predictor.display()

        # publish messages
        msg = conversions.cones_to_msg(cones)
        self.cone_publisher.publish(msg)
        print(cones)


def main(args=None):
    rclpy.init(args=args)

    stereo_node = LidarNode()

    rclpy.spin(stereo_node)

    stereo_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()