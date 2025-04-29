import rclpy
from rclpy.node import Node
from interfaces.msg import ConeArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

BEST_EFFORT_QOS_PROFILE = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT,
                         history = QoSHistoryPolicy.KEEP_LAST,
                         durability = QoSDurabilityPolicy.VOLATILE,
                         depth = 5)

class ConeInfoNode(Node):
    def __init__(self):
        super().__init__("cone_info_node")
        self.perc_cones_sub_ = self.create_subscription(ConeArray, "/perc_cones", self.perc_cones_callback, BEST_EFFORT_QOS_PROFILE)
    
    def perc_cones_callback(self, msg):
        print("Timestamp: ", msg.header.stamp)
        print("Num blue cones: ", len(msg.blue_cones))
        print("Num yellow cones: ", len(msg.yellow_cones))


def main():
    rclpy.init()
    node = ConeInfoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    return

if __name__ == "__main__":
    main()