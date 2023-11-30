# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

# ROS2 message types
from sensor_msgs.msg import Image, PointCloud2

# ROS2 msg to python datatype conversions
import perceptions.ros.utils.conversions as conv
from perceptions.ros.utils.topics import LEFT_IMAGE_TOPIC, RIGHT_IMAGE_TOPIC, XYZ_IMAGE_TOPIC, DEPTH_IMAGE_TOPIC, POINT_TOPIC

# perceptions Library visualization functions (for 3D data)
import perc22a.predictors.utils.lidar.visualization as vis
from perc22a.data.utils.DataType import DataType

# general imports
import cv2
import numpy as np

# configure QOS profile
BEST_EFFORT_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)

class DataNode(Node):

    def __init__(self, required_data=list(DataType), name="data_node", visualize=False):
        super().__init__(name)

        # subscribe to each piece of data that we want to collect on
        self.required_data = required_data
        self.visualize = visualize 

        if DataType.ZED_LEFT_COLOR in self.required_data:
            print("need zed left")
            self.left_color_subscriber = self.create_subscription(Image, LEFT_IMAGE_TOPIC, self.left_color_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        
        if DataType.ZED_RIGHT_COLOR in self.required_data:
            print("need zed right")
            self.right_color_subscriber = self.create_subscription(Image, RIGHT_IMAGE_TOPIC, self.right_color_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)

        if DataType.ZED_XYZ_IMG in self.required_data:
            print("need zed xyz")
            self.xyz_image_subscriber = self.create_subscription(Image, XYZ_IMAGE_TOPIC, self.xyz_image_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
            if self.visualize:
                self.xyz_image_window = vis.init_visualizer_window()

        if DataType.ZED_DEPTH_IMG in self.required_data:
            print("need zed depth")
            self.depth_subscriber = self.create_subscription(Image, DEPTH_IMAGE_TOPIC, self.depth_image_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)

        if DataType.HESAI_POINTCLOUD in self.required_data:
            print("need hesai")
            self.point_subscriber = self.create_subscription(PointCloud2, POINT_TOPIC, self.points_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
            if self.visualize:
                self.window = vis.init_visualizer_window()

        # define dictionary to store the data
        # TODO: convert data representation to DataInstance type
        self.data = {}
        self.left_color_str = "left_color"
        self.right_color_str = "right_color"
        self.xyz_image_str = "xyz_image"
        self.depth_image_str = "depth_image"
        self.points_str = "points"
        

    def got_all_data(self):
        # returns whether data node has all pieces of data
        return all([(data_type in self.required_data) for data_type in self.data.keys()])
    
    def left_color_callback(self, msg):
        self.data[self.left_color_str] = conv.img_to_npy(msg)

        if self.visualize:
            cv2.imshow("left", self.data[self.left_color_str])
            cv2.waitKey(1)

    def right_color_callback(self, msg):
        self.data[self.right_color_str] = conv.img_to_npy(msg)

        if self.visualize:
            cv2.imshow("right", self.data[self.right_color_str])
            cv2.waitKey(1)

    def xyz_image_callback(self, msg):
        self.data[self.xyz_image_str] =conv.img_to_npy(msg)

        if self.visualize:
            # display xyz_image as unstructured point cloud
            points = self.data[self.xyz_image_str][:, :, :3]
            points = points.reshape((-1, 3))
            points = points[:,[1,0,2]]
            points = points[~np.isnan(points)].reshape((-1, 3))
            points = points[points[:,2] > -1]

            vis.update_visualizer_window(self.xyz_image_window, points)

    def depth_image_callback(self, msg):
        self.data[self.depth_image_str] = conv.img_to_npy(msg)
        
        if self.visualize:
            cv2.imshow("depth", self.data[self.depth_image_str])

    def points_callback(self, msg):
        self.data[self.points_str] = conv.pointcloud2_to_npy(msg)

        if self.visualize:
            points = self.data[self.points_str][:, :3]
            points = points[:, [1, 0, 2]]
            points[:, 0] *= -1
            vis.update_visualizer_window(self.window, points[:,:3])


def main(args=None):
    rclpy.init(args=args)

    # enable absolutely ludicrous visualizations    
    data_node = DataNode(visualize=True)

    rclpy.spin(data_node)

    data_node.destroy_node()
    rclpy.shutdown()

    return

if __name__ == "__main__":
    main()