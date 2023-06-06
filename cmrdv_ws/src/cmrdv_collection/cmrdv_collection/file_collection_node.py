import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

# either convert to sensor msgs to some numpy msgs

import os
import numpy as np


DATA_DIR = "src/cmrdv_collection/data"


class FileCollectNode(Node):

    def __init__(self):
        super().__init__("FileCollectNode")

        # create publisher, dataset, and index
        self.dl = DataLoader(DATA_DIR)
        self.idx = 0
        self.size = len(self.dl)
        # self.publisher = self.create_publisher(msg_type=)
        self.timer = self.create_timer(1, self.push_data)
        pass

    def push_data(self):
        # load the appropriate data
        lidar, left, right, depth, pcd = self.dl[self.idx]
        self.idx = (self.idx + 1) % self.size

        # load into messages

        # publish
        print(f"{self.idx}", lidar.shape, left.shape, right.shape, depth.shape, pcd.shape)
        pass


class DataLoader:

    def __init__(self, path):
        self.cache = {} # MAYBE IMPLEMENT
        self.path = path
        self.files = sorted(os.listdir(path), key=lambda s: int(s[:-4]))
        self.nfiles = len(self.files)
        pass

    def __len__(self):
        return self.nfiles

    def __getitem__(self, key):
        file = os.path.join(self.path, self.files[key])
        data = np.load(file)
        return data["lidar_pcd"], data["zed_left"], data["zed_right"], data["zed_depth"], data["zed_pcd"]


def main(args=None):
    rclpy.init(args=args)

    collection = FileCollectNode()

    rclpy.spin(collection)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    collection.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    args = None
    main(args)
