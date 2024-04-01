# ROS2 imports
import rclpy

# for subscribing to sensor data
from perceptions.ros.utils.PredictNode import PredictNode

# for doing prediction on sensor data
from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor

class YOLOv5Node(PredictNode):

    def __init__(self, camera="zed2", debug=False, own_zed=None, publish_images=False):
        node_name = f"yolov5_{camera}_node"
        self.camera = camera
        
        # initialize attributes, then setup prediction node
        super().__init__(name=node_name, debug_flag=debug, time_flag=True, own_zed=own_zed, publish_images=publish_images)
        return

    def init_predictor(self):
        return YOLOv5Predictor(camera=self.camera)
    
def start_zed_node(args=None, camera="zed2", debug=False, own_zed=None, publish_images=False):
    rclpy.init(args=args)
    yolov5_node = YOLOv5Node(camera=camera, debug=debug, own_zed=own_zed, publish_images=publish_images)

    rclpy.spin(yolov5_node)

    yolov5_node.destroy_node()
    rclpy.shutdown()

def main_zed_debug(args=None):
    start_zed_node(args=args, camera="zed", debug=True)

def main_zed2_debug(args=None):
    start_zed_node(args=args, camera="zed2", debug=True)

def main_zed(args=None):
    start_zed_node(args=args, camera="zed", debug=False)

def main_zed2(args=None):
    start_zed_node(args=args, camera="zed2", debug=False)

def main_zed_own_debug(args=None):
    start_zed_node(args=args, camera="zed", debug=True, own_zed="zed")

def main_zed2_own_debug(args=None):
    start_zed_node(args=args, camera="zed2", debug=True, own_zed="zed2")

def main_zed_own(args=None):
    start_zed_node(args=args, camera="zed", debug=False, own_zed="zed")

def main_zed2_own(args=None):
    start_zed_node(args=args, camera="zed2", debug=False, own_zed="zed2")

def main_zed_own_publish(args=None):
    start_zed_node(args=args, camera="zed", debug=False, own_zed="zed", publish_images=True)

def main_zed2_own_publish(args=None):
    start_zed_node(args=args, camera="zed2", debug=False, own_zed="zed2", publish_images=True)



if __name__ == "__main__":
    main_zed2()