# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

# ROS2 message types
from sensor_msgs.msg import Image, PointCloud2

# ROS2 msg to python datatype conversions
import perceptions.utils.conversions as conv

# perceptions Library visualization functions (for 3D data)
import perc22a.predictors.utils.lidar.visualization as vis

# general imports
import cv2
import numpy as np

# configure QOS profile
BEST_EFFORT_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)

# setup the topic names that we are reading from
LEFT_IMAGE_TOPIC = "/zedsdk_left_color_image"
RIGHT_IMAGE_TOPIC = "/zedsdk_right_color_image"
XYZ_IMAGE_TOPIC = "/zedsdk_point_cloud_image"
DEPTH_IMAGE_TOPIC = "/zedsdk_depth_image"
POINT_TOPIC = "/lidar_points"


DEBUG = False

class DataNode(Node):

    def __init__(self, name="data_node"):
        super().__init__(name)

        if DEBUG:
            # setup point cloud visualization window
            self.window = vis.init_visualizer_window()
            self.xyz_image_window = vis.init_visualizer_window()

        # subscribe to each piece of data that we want to collect on
        self.left_color_subscriber = self.create_subscription(Image, LEFT_IMAGE_TOPIC, self.left_color_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.right_color_subscriber = self.create_subscription(Image, RIGHT_IMAGE_TOPIC, self.right_color_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.xyz_image_subscriber = self.create_subscription(Image, XYZ_IMAGE_TOPIC, self.xyz_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.depth_subscriber = self.create_subscription(Image, DEPTH_IMAGE_TOPIC, self.depthImage_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.point_subscriber = self.create_subscription(PointCloud2, POINT_TOPIC, self.point_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)

        # define varaibles to store the data
        self.left_color = None
        self.right_color = None
        self.xyz_image = None
        self.depth_image = None
        self.points = None
        

    def got_all_data(self):
        # returns whether data node has all pieces of data
        return self.left_color is not None and \
               self.right_color is not None and \
               self.xyz_image is not None and \
               self.depth_image is not None and \
               self.points is not None
    
    def left_color_callback(self, msg):
        self.left_color = conv.img_to_npy(msg)

        if DEBUG:
            cv2.imshow("left", self.left_color)
            cv2.waitKey(1)

    def right_color_callback(self, msg):
        self.right_color = conv.img_to_npy(msg)

        if DEBUG:
            cv2.imshow("right", self.right_color)
            cv2.waitKey(1)

    def xyz_callback(self, msg):
        self.xyz_image =conv.img_to_npy(msg)

        if DEBUG:
            # display xyz_image as unstructured point cloud
            points = self.xyz_image[:, :, :3]
            points = points.reshape((-1, 3))
            points = points[:,[1,0,2]]
            points = points[~np.isnan(points)].reshape((-1, 3))
            points = points[points[:,2] > -1]

            vis.update_visualizer_window(self.xyz_image_window, points)

    def depthImage_callback(self, msg):
        self.depth_image = conv.img_to_npy(msg)
        
        if DEBUG:
            cv2.imshow("depth", self.depth_image)

    def point_callback(self, msg):
        self.points = conv.pointcloud2_to_npy(msg)

        if DEBUG:
            vis.update_visualizer_window(self.window, self.points[:,:3])


def main(args=None):
    rclpy.init(args=args)

    # enable the debug flag for visualizations
    global DEBUG
    DEBUG = True

    data_node = DataNode()

    rclpy.spin(data_node)

    data_node.destroy_node()
    rclpy.shutdown()

    pass

if __name__ == "__main__":
    main()