import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2
from cmrdv_common.cmrdv_common.config import collection_config as cfg
from eufs_msgs.msg import ConeArrayWithCovariance
from cmrdv_interfaces.msg import SimDataFrame

class SimCollectionNode(Node):
    def __init__(self):
        super().__init__("SimCollectionNode")

        self.zed_left_sub = self.create_subscription() 
        Subscriber(self, Image, cfg.SIM_ZED_LEFT, qos_profile=cfg.BEST_EFFORT_QOS_PROFILE)
        self.zed_pts_sub = Subscriber(self, PointCloud2, cfg.SIM_ZED_POINTS, qos_profile=cfg.RELIABLE_QOS_PROFILE)
        self.vlp16_sub = Subscriber(self, PointCloud2, cfg.SIM_VLP16, qos_profile=cfg.BEST_EFFORT_QOS_PROFILE)
        self.gt_cones_sub = Subscriber(self, ConeArrayWithCovariance, cfg.SIM_GT_CONES, qos_profile=cfg.RELIABLE_QOS_PROFILE)

        self.approx_syncer = ApproximateTimeSynchronizer([self.gt_cones_sub, self.zed_left_sub, self.zed_pts_sub, self.vlp16_sub],
                                                          cfg.QUEUE_SIZE,
                                                          cfg.APPROX_DELAY)
        
        self.data_pub = self.create_publisher(SimDataFrame, cfg.DATA_TIME_SYNC, cfg.QUEUE_SIZE)
        self.approx_syncer.registerCallback(self.sync)

    def sync(self, gt_cones_msg, zed_left_msg, zed_pts_msg, vlp16_msg):
        print(zed_pts_msg.height, zed_pts_msg.width)
        sim_dataframe = SimDataFrame()
        sim_dataframe.zed_left_img = zed_left_msg
        sim_dataframe.zed_pts = zed_pts_msg
        sim_dataframe.vlp16_pts = vlp16_msg
        sim_dataframe.gt_cones = gt_cones_msg
        self.data_pub.publish(sim_dataframe)

def main(args=None):
    rclpy.init(args=args)

    collection = SimCollectionNode()

    rclpy.spin(collection)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    collection.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()