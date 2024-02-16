# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

# ROS2 message types
from sensor_msgs.msg import Image, PointCloud2
# from eufs_msgs.msg import DataFrame

# ROS2 msg to python datatype conversions
import perceptions.ros.utils.conversions as conv
from perceptions.topics import POINT_TOPIC, LEFT2_IMAGE_TOPIC, LEFT_IMAGE_TOPIC #, DATAFRAME_TOPIC

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

class LidarVisNode(Node):

    def __init__(self):
        super().__init__("lidar_vis_node")

        self.point_subscriber = self.create_subscription(PointCloud2, POINT_TOPIC, self.points_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.img2_subscriber = self.create_subscription(Image, LEFT2_IMAGE_TOPIC, self.left2_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.img_subscriber = self.create_subscription(Image, LEFT_IMAGE_TOPIC, self.left_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.window = vis.init_visualizer_window()

        self.update_fps = 30
        self.vis_update = self.create_timer(1 / self.update_fps, self.update_vis)

        self.points = None
        self.left = None
        self.left2 = None

                
    def points_callback(self, msg):
        pc = conv.pointcloud2_to_npy(msg)
        points = pc[:, :3]
        points = points[np.any(points != 0, axis=1)]

        points = points[:, [1, 0, 2]]
        points[:, 0] *= -1
        self.points = points
        # vis.update_visualizer_window(self.window, points[:,:3])
        print(f"{points.shape[0]} points")

    def left2_callback(self, msg):
        self.left2 = conv.img_to_npy(msg)

    def left_callback(self, msg):
        self.left = conv.img_to_npy(msg)
        
    def update_vis(self):
        if self.points is None or self.left is None or self.left2 is None:
            return

        vis.update_visualizer_window(self.window, self.points[:,:3])
        cv2.imshow("left", self.left)
        cv2.imshow("left2", self.left2)
        cv2.waitKey(1)
        return

def main(args=None):
    rclpy.init(args=args)

    # enable absolutely ludicrous visualizations    
    data_node = LidarVisNode()

    rclpy.spin(data_node)

    data_node.destroy_node()
    rclpy.shutdown()

    return

if __name__ == "__main__":
    main()
