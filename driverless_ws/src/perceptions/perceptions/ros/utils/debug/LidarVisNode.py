# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

# ROS2 message types
from sensor_msgs.msg import Image, PointCloud2
# from interfaces.msg import DataFrame

# ROS2 msg to python datatype conversions
import perceptions.ros.utils.conversions as conv
from perceptions.topics import POINT_TOPIC, POINT_2_TOPIC #, DATAFRAME_TOPIC

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

        self.points = None

        self.point_1_subscriber = self.create_subscription(PointCloud2, POINT_TOPIC, self.points_callback, qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.point_2_subscriber = self.create_subscription(PointCloud2, POINT_2_TOPIC, self.points_callback2, qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.window = vis.init_visualizer_window()

                
    def points_callback(self, msg):
        pc = conv.pointcloud2_to_npy(msg)
        points = pc[::10, :3] # downsample points
        # points = points[np.any(points != 0, axis=1)]

        tf_mat_left = np.array([[   0.76604444,    -0.64278764,    0.        ,  -0.18901      ],
                                [  0.64278764,    0.76604444,    0.        , 0.15407      ],
                                [   0.        ,    0.        ,    1.        ,    0.        ],
                                [   0.        ,    0.        ,    0.        ,    1.        ]])
        column = np.array([[1.0] for _ in range(len(points))])
        points = np.hstack((points, column))
        points = np.matmul(points, tf_mat_left.T)

        # multiply transformation matrix with points
        points = points[:, [1, 0, 2]]
        points[:, 0] *= -1
        if self.points is not None:
            vis.update_visualizer_window(self.window, np.vstack((self.points, points[:,:3])))
        self.points = points[:, :3]
            
        print(f"{points.shape[0]} L points")

    def points_callback2(self, msg):
        pc = conv.pointcloud2_to_npy(msg)
        points = pc[::10, :3]
        # points = points[np.any(points != 0, axis=1)]

        tf_mat_right = np.array([[  0.76604444,  0.64278764,   0.        , -0.16541      ],
                                 [  -0.64278764,   0.76604444,   0.        , -0.12595      ],
                                 [  0.        ,   0.        ,   1.        ,   0.        ],
                                 [  0.        ,   0.        ,   0.        ,   1.        ]])
        column = np.array([[1.0] for _ in range(len(points))])
        points = np.hstack((points, column))
        points = np.matmul(points, tf_mat_right.T)
        # points = np.transpose(np.matmul(tf_mat_right, np.transpose(points)))[:, :3]

        points = points[:, [1, 0, 2]]
        points[:, 0] *= -1
        if self.points is not None:
            vis.update_visualizer_window(self.window, np.vstack((self.points, points[:,:3])))
        self.points = points[:, :3]
        print(f"{points.shape[0]} points")

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
