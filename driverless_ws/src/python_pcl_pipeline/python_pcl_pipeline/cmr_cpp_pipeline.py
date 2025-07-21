import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt

from .helper import *

from sensor_msgs.msg import PointCloud2

print(np.__version__)


def pointcloud2_to_npy(pc2_msg: PointCloud2):
    points_raw = point_cloud2_to_dict(pc2_msg)
    # print(type(points_raw['xyz']))
    # points_arr = np.zeros((points_raw['xyz'].shape[0], 3))

    # points_arr[:, 0] = points_raw['x'].reshape(-1)
    # points_arr[:, 1] = points_raw['y'].reshape(-1)
    # points_arr[:, 2] = points_raw['z'].reshape(-1)
    # points_arr[:,3] = points_raw['intensity'].reshape(-1)
    # points_arr[:,4] = points_raw['ring'].reshape(-1)
    # points_arr[:,5] = points_raw['timestamp'].reshape(-1)

    # return points_arr

    return points_raw['xyz'], points_raw['intensity']


def npy_to_pointcloud2(pc, intensities):

    # --- STEP 1: SANITY CHECK SNIPPET ---
    print("--- INPUT DATA SANITY CHECK ---")

    xyz_data = pc
    intensity_data = intensities

    if xyz_data is not None:
        print(f"XYZ data shape: {xyz_data.shape}, dtype: {xyz_data.dtype}")
        # Print min/max/mean for each axis to ensure points are where you expect
        print(f"  X axis: min={np.min(xyz_data[:, 0]):.2f}, max={np.max(xyz_data[:, 0]):.2f}, mean={np.mean(xyz_data[:, 0]):.2f}")
        print(f"  Y axis: min={np.min(xyz_data[:, 1]):.2f}, max={np.max(xyz_data[:, 1]):.2f}, mean={np.mean(xyz_data[:, 1]):.2f}")
        print(f"  Z axis: min={np.min(xyz_data[:, 2]):.2f}, max={np.max(xyz_data[:, 2]):.2f}, mean={np.mean(xyz_data[:, 2]):.2f}")
    else:
        print("XYZ data not found!")

    if intensity_data is not None:
        print(f"Intensity data shape: {intensity_data.shape}, dtype: {intensity_data.dtype}")
        print(f"  Intensity: min={np.min(intensity_data):.2f}, max={np.max(intensity_data):.2f}, mean={np.mean(intensity_data):.2f}")
    else:
        print("Intensity data not found.")
    print("---------------------------------\n")

    pc_msg = dict_to_point_cloud2({'xyz': pc, 'intensity': intensities})
    return pc_msg


class CMRCPPPipelineNode(Node):

    def __init__(self):
        super().__init__('cmr_cpp_pipeline_node')

        self.sub = self.create_subscription(PointCloud2, 'lidar_points', self.point_callback, 10)
        self.pub = self.create_publisher(PointCloud2, 'filtered_lidar_points', 10)

        self.alpha = .1
        self.num_bins = 3
        self.height_threshold = .1

        self.epsilon = .2
        self.min_points = 3

        self.epsilon2 = 3
        self.min_points2 = 3
        
        self.epsilon3 = 3
        self.min_points3 = 3

    def gnc(self, cloud, intensities):
        return grace_and_conrad(cloud, intensities, self.alpha, self.num_bins, self.height_threshold)

    def dbs(self, cloud):
        return dbscan_optimized(cloud, self.epsilon, self.min_points)

    def dbs2(self, cloud):
        return dbscan2_optimized_and_filtered(cloud, self.epsilon2, self.min_points2)
    
    def dbs3(self, cloud, intensities):
        return dbscan3_get_each_cone_cluster(cloud, intensities, self.epsilon3, self.min_points3)
    
    def to_grid(self, cloud, intensities):
        return cone_cluster_to_intensity_grid(cloud, intensities)

    def point_callback(self, cloud):

        points, intensities = pointcloud2_to_npy(cloud)

        dist = np.linalg.norm(points, axis=1)
        points = points[dist > .01]
        intensities = intensities[dist > .01]

        points, intensities = self.gnc(points, intensities)
        print(points.shape, intensities.shape)
        # points = self.dbs(points)
        # points = self.dbs2(points)
        cone_clusters = self.dbs3(points, intensities)
        print(f'    --> Found {len(cone_clusters)} potential cones')
        
        cone_points_pub = []
        last_grid = None
        
        for i, (points, intensities) in enumerate(cone_clusters):
            grid = self.to_grid(points, intensities)
            # TODO
            # color = self.AMZ_model.predict(grid)
            color = "HA --- GOTEM"
            print(f'    --> Cone #{i}: Processed {points.shape[0]} points into a {grid.shape} grid, with color: {color}')
            
            cone_points_pub.append(points)
            last_grid = grid
        
        # if last_grid is not None:
        #     # Offload this to rviz or foxglove
        #     plt.figure(i + 1)
        #     plt.clf()
        #     plt.imshow(grid, cmap='gray', vmin=0, vmax=1)
        #     plt.title(f"Intensity Grid for Cone #{i+1}")
        #     plt.show(block=False)
        #     plt.pause(0.1)
            
        points = np.vstack(cone_points_pub)
        intensities = np.ones(points.shape[0])

        # print(intensities.min(), intensities.max())

        # print(points.shape, intensities.shape)

        # np.savez(
        #     r'/home/aryalohia/CMR/25a/ros2_ws/src/rosbag_processing/rosbag_processing/saves/scene.npz',
        #     np.hstack((points, intensities[:, np.newaxis]))
        # )

        # new_cloud = npy_to_pointcloud2(points, np.ones(shape=points.shape[0]))
        new_cloud = npy_to_pointcloud2(points, intensities)
        new_cloud.header = cloud.header
        # new_cloud.height = cloud.height
        # new_cloud.width = points.shape[0]
        # # new_cloud.fields = cloud.fields
        # # new_cloud.is_dense = cloud.is_dense
        # new_cloud.is_bigendian = cloud.is_bigendian
        # new_cloud.point_step = 14
        # new_cloud.row_step = new_cloud.point_step * new_cloud.width

        self.pub.publish(new_cloud)


def main(args=None):
    rclpy.init(args=args)
    node = CMRCPPPipelineNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
