# ROS2 imports
import rclpy

# for subscribing to sensor data
from perceptions.utils.DataNode import DataNode

# for doing prediction on sensor data
from perc22a.predictors import StereoPredictor

class StereoNode(DataNode):

    def __init__(self):
        super().__init__(name="stereo_node")

        # do prediction on a timer
        self.interval = 1
        self.predict_timer = self.create_timer(self.interval, self.predict_callback)

        # create predictor
        self.model_name = 'ultralytics/yolov5'
        self.param_path = '/home/dale/driverless-packages/PerceptionsLibrary22a/perc22a/predictors/stereo/model_params.pt'
        self.predictor = StereoPredictor(self.model_name, self.param_path)

    def predict_callback(self):
        if not self.got_all_data():
            self.get_logger().warn("Not got all data")
            return
    
        # otherwise, do prediction on data
        data = {
            "left_color": self.left_color,
            "xyz_image": self.xyz_image,
        }
        cones = self.predictor.predict(data)

        self.predictor.display()
        print("cones", cones)


def main(args=None):
    rclpy.init(args=args)

    stereo_node = StereoNode()

    rclpy.spin(stereo_node)

    stereo_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()