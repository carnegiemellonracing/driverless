import rclpy
from rclpy.node import Node

# include DataNode for subscribing to data
from perceptions.ros.utils.DataNode import DataNode

# required datatypes in case necessary to modify
from perc22a.data.utils.DataType import DataType

# file path manipulation and creating directories
from pathlib import Path
import shutil
import os

# use numpy to save files
import numpy as np

# set to determine what folder to create (find in ~/driverless/driverless_ws/<FOLDER_NAME>)
# DO NOT MAKE "src", "build", "install", or "log"
FOLDER_NAME = "tt-4-18-eleventh"

# define path to data directory
WS_DIR = Path(__file__).parents[3]
DATA_DIR = os.path.join(WS_DIR, FOLDER_NAME)

class FileNode(DataNode):

    def __init__(self):
        super().__init__(name="file_node", required_data=[DataType.HESAI_POINTCLOUD])

        # create timer for saving on interval
        self.interval = 0.1
        self.save_timer = self.create_timer(self.interval, self.save_callback)
        self.save_instance = 0

        # setup empty
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
            print(f"Deleting existing {FOLDER_NAME} to reset data saving")
        os.mkdir(DATA_DIR)

    def get_datafile_name(self):
        return f"instance-{self.save_instance}.npz"

    def save_callback(self):
        if not self.got_all_data():
            self.get_logger().warn("Not got all data")
            return
        
        datafile_name = self.get_datafile_name()
        print(f"Saving instance {self.save_instance} @ {os.path.join(FOLDER_NAME, datafile_name)}")

        # self.data updated by DataNode subscribers
        filepath = os.path.join(DATA_DIR, datafile_name)
        self.data.save(filepath)

        # update instance value
        self.save_instance += 1
        return

def main(args=None):
    rclpy.init(args=args)

    file_node = FileNode()

    rclpy.spin(file_node)

    file_node.destroy_node()
    rclpy.shutdown()

    return

if __name__ == "__main__":
    main()