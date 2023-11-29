# ROS2 imports
import rclpy

# for subscribing to sensor data
from perceptions.DataNode import DataNode

# for converting predictor output to cone message type
from eufs_msgs.msg import ConeArray
import perceptions.conversions as conversions

# for doing prediction on sensor data
from perc22a.predictors.stereo import StereoPredictor

NODE_NAME = "stereo_node"

class StereoNode(DataNode):

    def __init__(self):
        super().__init__(name=NODE_NAME)

        # do prediction on a timer
        # TODO: figure out what the best way is to deal with this?
        self.interval = 0.5
        self.predict_timer = self.create_timer(self.interval, self.predict_callback)

        # create publisher
        self.cone_topic = f"/{NODE_NAME}_cones"
        self.qos_profile = 10
        self.cone_publisher = self.create_publisher(ConeArray, self.cone_topic, self.qos_profile)

        # create predictor
        self.model_name = 'ultralytics/yolov5'
        self.param_path = '/home/dale/driverless-packages/PerceptionsLibrary22a/perc22a/predictors/stereo/model_params.pt'
        self.predictor = StereoPredictor(self.model_name, self.param_path)

    def predict_callback(self):
        if not self.got_all_data():
            self.get_logger().warn("Not got all data")
            return
    
        # otherwise, do prediction on data and display
        data = {
            "left_color": self.left_color,
            "xyz_image": self.xyz_image,
        }
        cones = self.predictor.predict(data)
        self.predictor.display()

        # publish message
        msg = conversions.cones_to_msg(cones)
        self.cone_publisher.publish(msg)

        print(cones)


def main(args=None):
    rclpy.init(args=args)

    stereo_node = StereoNode()

    rclpy.spin(stereo_node)

    stereo_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
