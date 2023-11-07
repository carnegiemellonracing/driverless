import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import Point
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config import collection_config as collection_cfg
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config import perceptions_config as perceptions_cfg
from cmrdv_ws.src.cmrdv_perceptions.lidar.lidar import LidarPredictor
import cmrdv_ws.src.cmrdv_perceptions.lidar.visualization as lidar_vis

from cmrdv_ws.src.cmrdv_perceptions.utils.utils import np2points
from cmrdv_ws.src.cmrdv_common.cmrdv_common.conversions import pointcloud2_to_array
from cmrdv_interfaces.msg import DataFrame, SimDataFrame, ConeList

import numpy as np


class LidarNode(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.declare_parameter('use_simulated_data', False)

        self.predictor = LidarPredictor()
        self.sim = self.get_parameter('use_simulated_data').get_parameter_value().bool_value

        self.subscription = self.create_subscription(msg_type=SimDataFrame if self.sim else DataFrame,
                                                     topic=collection_cfg.SIM_DATA_TIME_SYNC if self.sim else collection_cfg.DATA_TIME_SYNC,
                                                     callback=self.inference,
                                                     qos_profile=collection_cfg.BEST_EFFORT_QOS_PROFILE
                                                     )
        self.publisher = self.create_publisher(msg_type=ConeList,
                                               topic=perceptions_cfg.LIDAR_OUT,
                                               qos_profile=collection_cfg.BEST_EFFORT_QOS_PROFILE)

        # self.subscription  # prevent unused variable warning
        self.vis = lidar_vis.init_visualizer_window()

    def reformat_pointcloud(self, pc):
        pc_raw = pointcloud2_to_array(pc)

        xs = pc_raw['x'].reshape(-1)
        ys = pc_raw['y'].reshape(-1)
        zs = pc_raw['z'].reshape(-1)

        idxs = ~np.isnan(xs)

        xs = xs[idxs].reshape((-1,1))
        ys = ys[idxs].reshape((-1,1))
        zs = zs[idxs].reshape((-1,1))

        pc_data = np.hstack([xs, ys, zs])

        return pc_data

    def fix_coordinates(self, pc):
        pc = pc[:, [1, 0, 2]]
        pc[:,0] *= -1
        return pc

    def inference(self, dataframe):

        # read data from message and unpack point cloud
        pc = dataframe.vlp16_pts

        pc_data = self.reformat_pointcloud(pc)
        pc_data = self.fix_coordinates(pc_data)

        lidar_vis.update_visualizer_window(self.vis, pc_data)

        blue_cones, yellow_cones, orange_cones = self.predictor.predict(pc_data)
        import pdb; pdb.set_trace()

        # publish message to appropriate output
        #msg = ConeList()
        #msg.blue_cones = np2points(blue_cones)
        #msg.yellow_cones = np2points(yellow_cones)
        #msg.orange_cones = np2points(orange_cones)

        #self.publisher.publish(msg)

        pass


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = LidarNode()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
