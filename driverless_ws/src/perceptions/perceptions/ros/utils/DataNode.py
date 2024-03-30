# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

# ROS2 message types
from sensor_msgs.msg import Image, PointCloud2
# from interfaces.msg import DataFrame

# ROS2 msg to python datatype conversions
import perceptions.ros.utils.conversions as conv
from perceptions.topics import LEFT_IMAGE_TOPIC, RIGHT_IMAGE_TOPIC, XYZ_IMAGE_TOPIC, DEPTH_IMAGE_TOPIC, POINT_TOPIC #, DATAFRAME_TOPIC
from perceptions.topics import LEFT2_IMAGE_TOPIC, RIGHT2_IMAGE_TOPIC, XYZ2_IMAGE_TOPIC, DEPTH2_IMAGE_TOPIC
from perceptions.zed import ZEDSDK

# perceptions Library visualization functions (for 3D data)
import perc22a.predictors.utils.lidar.visualization as vis
from perc22a.data.utils.DataType import DataType
from perc22a.data.utils.DataInstance import DataInstance

# general imports
import cv2
import numpy as np

PUBLISH_FPS = 15

# configure QOS profile
BEST_EFFORT_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)
RELIABLE_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.RELIABLE,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)

# allowing subscriptions to all sensor datatypes
ALL_DATA_TYPES = [d for d in DataType]

class DataNode(Node):

    def __init__(self, required_data=ALL_DATA_TYPES, name="data_node", visualize=False, own_zed=None):

        super().__init__(name)

        assert(own_zed == None or own_zed == "zed" or own_zed == "zed2" or own_zed == "both")

        if own_zed == "zed" or own_zed == "both":
            self.zed = ZEDSDK(serial_num=15080)
            self.zed.open()
            self.data_syncer = self.create_timer(1/PUBLISH_FPS, self.update_zed_data)
        if own_zed == "zed2" or own_zed == "both":
            self.zed2 = ZEDSDK(serial_num=27680008)
            self.zed2.open()
            self.data_syncer = self.create_timer(1/PUBLISH_FPS, self.update_zed2_data)

        # subscribe to each piece of data that we want to collect on
        self.required_data = required_data
        self.visualize = visualize

        # define dictionary to store the data
        self.data = DataInstance(required_data)


        if DataType.ZED_LEFT_COLOR in self.required_data and (own_zed == "zed2" or own_zed == None):
            self.left_color_subscriber = self.create_subscription(Image, LEFT_IMAGE_TOPIC, self.left_color_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        
        if DataType.ZED2_LEFT_COLOR in self.required_data and (own_zed == "zed" or own_zed == None):
            self.left2_color_subscriber = self.create_subscription(Image, LEFT2_IMAGE_TOPIC, self.left2_color_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        
        # if DataType.ZED_RIGHT_COLOR in self.required_data:
        #     self.right_color_subscriber = self.create_subscription(Image, RIGHT_IMAGE_TOPIC, self.right_color_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)

        if DataType.ZED_XYZ_IMG in self.required_data and (own_zed == "zed2" or own_zed == None):
            self.xyz_image_subscriber = self.create_subscription(Image, XYZ_IMAGE_TOPIC, self.xyz_image_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
            if self.visualize:
                self.xyz_image_window = vis.init_visualizer_window()
                
        if DataType.ZED2_XYZ_IMG in self.required_data and (own_zed == "zed" or own_zed == None):
            self.xyz2_image_subscriber = self.create_subscription(Image, XYZ2_IMAGE_TOPIC, self.xyz2_image_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
            if self.visualize:
                self.xyz2_image_window = vis.init_visualizer_window()

        # if DataType.ZED_DEPTH_IMG in self.required_data:
        #     self.depth_subscriber = self.create_subscription(Image, DEPTH_IMAGE_TOPIC, self.depth_image_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)

        if DataType.HESAI_POINTCLOUD in self.required_data:
            self.point_subscriber = self.create_subscription(PointCloud2, POINT_TOPIC, self.points_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
            if self.visualize:
                self.window = vis.init_visualizer_window()

        # if DataType.DATAFRAME in self.required_data:
        #     self.dataframe_subscriber = self.create_subscription(DataFrame, DATAFRAME_TOPIC, self.dataframe_callback, qos_profile=RELIABLE_QOS_PROFILE)
        #     if self.visualize:
        #         self.window = vis.init_visualizer_window()
                
    def flush(self):
        # flushes data so that all required data must be collected again
        self.data = DataInstance(self.required_data)

    def got_all_data(self):
        # returns whether data node has all pieces of data
        return self.data.have_all_data()
    
    def update_zed_data(self):
        left, right, depth, xyz = self.zed.grab_data()
        self.data[DataType.ZED_LEFT_COLOR] = left
        self.data[DataType.ZED_XYZ_IMG] = xyz

    def update_zed2_data(self):
        left, right, depth, xyz = self.zed2.grab_data()
        self.data[DataType.ZED2_LEFT_COLOR] = left
        self.data[DataType.ZED2_XYZ_IMG] = xyz
    
    def left_color_callback(self, msg):
        self.data[DataType.ZED_LEFT_COLOR] = conv.img_to_npy(msg)

        if self.visualize:
            cv2.imshow("left", self.data[DataType.ZED_LEFT_COLOR])
            cv2.waitKey(1)
            
    def left2_color_callback(self, msg):
        self.data[DataType.ZED2_LEFT_COLOR] = conv.img_to_npy(msg)

        if self.visualize:
            cv2.imshow("left2", self.data[DataType.ZED2_LEFT_COLOR])
            cv2.waitKey(1)

    def right_color_callback(self, msg):
        self.data[DataType.ZED_RIGHT_COLOR] = conv.img_to_npy(msg)

        if self.visualize:
            cv2.imshow("right", self.data[DataType.ZED_RIGHT_COLOR])
            cv2.waitKey(1)

    def xyz_image_callback(self, msg):
        self.data[DataType.ZED_XYZ_IMG] = conv.img_to_npy(msg)

        if self.visualize:
            # display xyz_image as unstructured point cloud
            points = self.data[DataType.ZED_XYZ_IMG][:, :, :3]
            points = points.reshape((-1, 3))
            points = points[:,[1,0,2]]
            points = points[~np.isnan(points)].reshape((-1, 3))
            points = points[points[:,2] > -1]

            vis.update_visualizer_window(self.xyz_image_window, points)
            
    def xyz2_image_callback(self, msg):
        self.data[DataType.ZED2_XYZ_IMG] = conv.img_to_npy(msg)

        if self.visualize:
            # display xyz2_image as unstructured point cloud
            points = self.data[DataType.ZED2_XYZ_IMG][:, :, :3]
            points = points.reshape((-1, 3))
            points = points[:,[1,0,2]]
            points = points[~np.isnan(points)].reshape((-1, 3))
            points = points[points[:,2] > -1]

            vis.update_visualizer_window(self.xyz2_image_window, points)

    def depth_image_callback(self, msg):
        self.data[DataType.ZED_DEPTH_IMG] = conv.img_to_npy(msg)
        
        if self.visualize:
            cv2.imshow("depth", self.data[DataType.ZED_DEPTH_IMG])

    def points_callback(self, msg):
        self.data[DataType.HESAI_POINTCLOUD] = conv.pointcloud2_to_npy(msg)

        if self.visualize:
            points = self.data[DataType.HESAI_POINTCLOUD][:, :3]
            points = points[:, [1, 0, 2]]
            points[:, 0] *= -1
            vis.update_visualizer_window(self.window, points[:,:3])
    
    def dataframe_callback(self, msg):
        self.data[DataType.HESAI_POINTCLOUD] = conv.pointcloud2_to_npy(msg.pointcloud_msg)
        self.data[DataType.ZED_LEFT_COLOR] = conv.img_to_npy(msg.image_msg)

        if self.visualize:
            points = self.data[DataType.HESAI_POINTCLOUD][:, :3]
            points = points[:, [1, 0, 2]]
            points[:, 0] *= -1
            elapsed_time = vis.update_visualizer_window(None, points[:,:3])
            print(f"Vis Elaped Time: {elapsed_time}ms")
            cv2.imshow("left", self.data[DataType.ZED_LEFT_COLOR])
            cv2.waitKey(int(elapsed_time))

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
