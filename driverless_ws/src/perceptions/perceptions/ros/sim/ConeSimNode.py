# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

# topic names
from perceptions.topics import PERC_CONE_TOPIC

# cone datatype for ROS and perc22a
from interfaces.msg import ConeArray
from perc22a.predictors.utils.cones import Cones
import perceptions.ros.utils.conversions as conv

# Cone Merger and pipeline enum type
from perc22a.utils.ConeSim import ConeSim

# perceptions Library visualization functions (for 2D data)
from perc22a.predictors.utils.vis.Vis2D import Vis2D



# configure QOS profile
BEST_EFFORT_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)
RELIABLE_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.RELIABLE,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)

CONE_NODE_NAME = "sim_cone_node"
PUBLISH_FPS = 10
VIS_UPDATE_FPS = 25
MAX_ZED_CONE_RANGE = 12.5

class ConeSimNode(Node):

    def __init__(self, debug=True):
        super().__init__(CONE_NODE_NAME)

        # save our merger for merging cone outputs    
        self.cone_sim = ConeSim(period=5)

        # initialize cone publisher
        self.publish_timer = self.create_timer(1/PUBLISH_FPS, self.publish_cones)
        self.cone_publisher = self.create_publisher(ConeArray, PERC_CONE_TOPIC, qos_profile=RELIABLE_QOS_PROFILE)

        # deubgging mode visualizer
        if debug:
            self.vis2D = Vis2D()
            self.display_timer = self.create_timer(1/VIS_UPDATE_FPS, self.update_vis)

        # if debugging, initialize visualizer
        self.debug = debug

        return

    def update_vis(self):
        # update and interact with vis
        self.vis2D.update()
        return

    def publish_cones(self):

        cones = self.cone_sim.get_cones()

        # update visualizer
        if self.debug:
            self.vis2D.set_cones(cones)

        # publish cones
        print(f"Published {len(cones)} cones")
        msg = conv.cones_to_msg(cones)
        self.cone_publisher.publish(msg)
        
        return

def start_node(args, debug=False):
    rclpy.init(args=args)

    cone_node = ConeSimNode(debug=debug)

    rclpy.spin(cone_node)

    cone_node.destroy_node()
    rclpy.shutdown()

    return

# TODO: decide which policy is best and call it from main() (default to zed)
def main(args=None):
    start_node(args, debug=False)
    
def main_debug(args=None):
    start_node(args, debug=True)

if __name__ == "__main__":
    main()
