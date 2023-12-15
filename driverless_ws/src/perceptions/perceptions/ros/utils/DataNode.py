# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from eufs_msgs.msg import DataFrame

# ROS2 message types
from sensor_msgs.msg import Image, PointCloud2

# ROS2 msg to python datatype conversions
import perceptions.ros.utils.conversions as conv

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

RELIABLE_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.RELIABLE,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)

# setup the topic names that we are reading from
LEFT_IMAGE_TOPIC = "/zedsdk_left_color_image"
RIGHT_IMAGE_TOPIC = "/zedsdk_right_color_image"
XYZ_IMAGE_TOPIC = "/zedsdk_point_cloud_image"
DEPTH_IMAGE_TOPIC = "/zedsdk_depth_image"
POINT_TOPIC = "/lidar_points"
DATAFRAME_TOPIC = "/DataFrame"

DEBUG = False

class DataNode(Node):

    def __init__(self, name="data_node"):
        super().__init__(name)

        if DEBUG:
            # setup point cloud visualization window
            self.window = vis.init_visualizer_window()
            self.xyz_image_window = vis.init_visualizer_window()

        # subscribe to each piece of data that we want to collect on
        # self.left_color_subscriber = self.create_subscription(Image, LEFT_IMAGE_TOPIC, self.left_color_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        # self.right_color_subscriber = self.create_subscription(Image, RIGHT_IMAGE_TOPIC, self.right_color_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        # self.xyz_image_subscriber = self.create_subscription(Image, XYZ_IMAGE_TOPIC, self.xyz_image_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        # self.depth_subscriber = self.create_subscription(Image, DEPTH_IMAGE_TOPIC, self.depth_image_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        # self.point_subscriber = self.create_subscription(PointCloud2, POINT_TOPIC, self.points_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.dataframe_subscriber = self.create_subscription(DataFrame, DATAFRAME_TOPIC, self.dataframe_callback, qos_profile=RELIABLE_QOS_PROFILE)

        # define dictionary to store the data
        self.data = {}

        # create key strings associated with data (eventually make same as topic names)
        self.left_color_str = "left_color"
        self.right_color_str = "right_color"
        self.xyz_image_str = "xyz_image"
        self.depth_image_str = "depth_image"
        self.points_str = "points"
        self.imu_data = "imu_data"
        self.lin_vel_str = "lin_vel"

    def got_all_data(self):
        # returns whether data node has all pieces of data
        # return self.left_color_str in self.data and \
        #        self.right_color_str in self.data and \
        #        self.xyz_image_str in self.data and \
        #        self.depth_image_str in self.data and \
        #        self.points_str in self.data
        return self.left_color_str in self.data and \
               self.imu_data in self.data and \
               self.lin_vel_str in self.data and \
               self.xyz_image_str in self.data
    
    def left_color_callback(self, msg):
        self.data[self.left_color_str] = conv.img_to_npy(msg)

        if DEBUG:
            cv2.imshow("left", self.data[self.left_color_str])
            cv2.waitKey(1)

    def right_color_callback(self, msg):
        self.data[self.right_color_str] = conv.img_to_npy(msg)

        if DEBUG:
            cv2.imshow("right", self.data[self.right_color_str])
            cv2.waitKey(1)

    def xyz_image_callback(self, msg):
        self.data[self.xyz_image_str] = conv.img_to_npy(msg)

        if DEBUG:
            # display xyz_image as unstructured point cloud
            points = self.data[self.xyz_image_str][:, :, :3]
            points = points.reshape((-1, 3))
            points = points[:,[1,0,2]]
            points = points[~np.isnan(points)].reshape((-1, 3))
            points = points[points[:,2] > -1]

            vis.update_visualizer_window(self.xyz_image_window, points)

    def depth_image_callback(self, msg):
        self.data[self.depth_image_str] = conv.img_to_npy(msg)
        
        if DEBUG:
            cv2.imshow("depth", self.data[self.depth_image_str])

    def points_callback(self, msg):
        self.data[self.points_str] = conv.pointcloud2_to_npy(msg)

        if DEBUG:
            points = self.data[self.points_str][:, :3]
            points = points[:, [1, 0, 2]]
            points[:, 0] *= -1
            vis.update_visualizer_window(None, points[:,:3])
            # vis.update_visualizer_window(self.window, points[:,:3])


    def dataframe_callback(self, msg):
        self.data[self.left_color_str] = conv.img_to_npy(msg.image_msg)
        self.data[self.xyz_image_str] = conv.img_to_npy(msg.xyz_msg)
        self.data[self.imu_data] = msg.imu_data
        self.data[self.lin_vel_str] = msg.imu_linear_velocity

def main(args=None):
    rclpy.init(args=args)

    # enable the debug flag for sexy visualizations
    global DEBUG
    DEBUG = True

    data_node = DataNode()

    rclpy.spin(data_node)

    data_node.destroy_node()
    rclpy.shutdown()

    return

if __name__ == "__main__":
    main()