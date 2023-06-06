import rclpy
from rclpy.node import Node
import cmrdv_ws.src.cmrdv_common.cmrdv_common.config.perceptions_config as cfg_perceptions
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.collection_config import BEST_EFFORT_QOS_PROFILE
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.planning_config import *
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.common_config import *
from eufs_msgs.msg import ConeArrayWithCovariance
from cmrdv_interfaces.msg import CarROT, ConeList, Points

# python and visualizer imports
import numpy as np
from cmrdv_ws.src.cmrdv_perceptions.utils.cvvis import SimCVVis

class SimVisNode(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.gt_cones_sub = self.create_subscription(
            ConeArrayWithCovariance,
            SIM_GT_CONES,
            self.save_gt_cones,
            QoSProfile(reliability = QoSReliabilityPolicy.RELIABLE,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)
        )
        
        self.cones_sub = self.create_subscription(
                ConeList,
                CONE_DATA_TOPIC,
                self.save_cones,
                qos_profile=BEST_EFFORT_QOS_PROFILE
        )

        self.carrot_sub = self.create_subscription(
                CarROT,
                CARROT_TOPIC,
                self.save_carrot,
                qos_profile=BEST_EFFORT_QOS_PROFILE
        )

        self.spline_pub = self.create_subscription(
                Points,
                SPLINE_TOPIC,
                self.save_spline,
                qos_profile=BEST_EFFORT_QOS_PROFILE
        )
        self.subscription  # prevent unused variable warning

     # init cvvis
        self.simvis = SimCVVis()
        self.simvis.start()

        # create timer callback to udpdate cvvis
        hz = 8
        self.timer = self.create_timer(1 / hz, self.update_callback)
        
        # default data values
        self.spline_points = np.array([[0,0]]).reshape((1,2))
        self.cones = np.array([]).reshape((0,3))
        self.x = 0
        self.y = 0
        self.yaw = 0
        self.curvature = 0

        print("started visualizer")
    def save_gt_cones(self, msg):
        arr = []
        for blue_cone in msg.blue_cones:
            arr.append([blue_cone.point.x, blue_cone.point.z, 1])

        for yellow_cone in msg.yellow_cones:
            arr.append([yellow_cone.point.x, yellow_cone.point.z, 1])

        for orange_cone in msg.orange_cones:
            arr.append([orange_cone.point.x, orange_cone.point.z, 1])

        for big_orange_cone in msg.big_orange_cones:
            arr.append([big_orange_cone.point.x, big_orange_cone.point.z, 1])

        for unknown_cone in msg.unknown_cones:
            arr.append([unknown_cone.point.x, unknown_cone.point.z, 1])
        self.gt_cones = np.array(arr)

    def save_cones(self, msg):
        # format the cones and store 
        arr = []
        arr += [[c.x, c.y, cfg_perceptions.COLORS.BLUE.value] for c in msg.blue_cones]
        arr += [[c.x, c.y, cfg_perceptions.COLORS.YELLOW.value] for c in msg.yellow_cones]
        arr += [[c.x, c.y, cfg_perceptions.COLORS.ORANGE.value] for c in msg.orange_cones]
        self.cones = np.array(arr)

    def save_carrot(self, msg):
        self.x, self.y = msg.x, msg.y
        self.yaw, self.curvature = msg.yaw, msg.curvature
    
    def save_spline(self, msg):
        points = msg.points
        spline = []
        for p in points:
            spline.append([p.x, p.y])
        self.spline_points = np.array(spline)

    def update_callback(self):
        carrot = np.array([[self.x, self.y]])
        # self.get_logger().info(str(self.cones[:, :2]))
        # self.get_logger().info(str(self.cones[:, :2].shape))
        # self.get_logger().info(str(self.spline_points.shape))
        self.simvis.sim_update(self.gt_cones, self.cones, self.spline_points, carrot, self.yaw, self.curvature)

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = SimVisNode()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()