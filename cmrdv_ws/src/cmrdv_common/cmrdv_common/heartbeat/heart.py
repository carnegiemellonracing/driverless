import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from cmrdv_interfaces.msg import Heartbeat
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.collection_config import RELIABLE_QOS_PROFILE
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.common_config import GLOBAL_TIMEOUT_SEC
from cmrdv_ws.src.cmrdv_common.cmrdv_common.CAN.can_types import ALIVE_STATE, WARNING_STATE, ERROR_STATE, TX10HZ_PERIOD_S

# from cmrdv_common.config.collection_config import RELIABLE_QOS_PROFILE
# from cmrdv_common.config.common_config import GLOBAL_TIMEOUT_SEC
# from cmrdv_common.CAN.can_types import ALIVE_STATE, WARNING_STATE, ERROR_STATE, TX10HZ_PERIOD_S


class HeartbeatNode(Node):
    """
    Base class for each vital node to inherit. Creates a publisher for a /`{node}_status` topic and 
    subscribes to the /global_heartbeat topic. Default timer period is 10hz for updating the nodes status
    and reading the global status

    Parameters
    ----------
    node_name : str
        Node name found on the topic, `{node}_status`
    timer_period : float
        Frequency to publish status to system
    """
    def __init__(self, node_name, timer_period=TX10HZ_PERIOD_S):
        super().__init__(node_name)
        self.node_name = node_name

        self.time = self.get_clock().now()
        self.timer = self.create_timer(timer_period, self.node_status_pub)

        self.threshold = rclpy.time.Duration(seconds=(4 * timer_period))

        # TODO: perhaps change node status to correspond to CAN status
        self.node_status = ALIVE_STATE
        self.global_status = ALIVE_STATE

        self.status_publisher = self.create_publisher(
            Heartbeat,
            f"{node_name}_status",
            RELIABLE_QOS_PROFILE
        )
        self.global_heartbeat = self.create_subscription(
            Heartbeat, 
            "global_heartbeat", 
            self.update_status, 
            RELIABLE_QOS_PROFILE
        )

        # TODO: why use global timeout sec and not check at 10hz?
        self.last_global_heartbeat = self.get_clock().now()
        self.check_global_timeout = self.create_timer(
            GLOBAL_TIMEOUT_SEC, 
            self.check_global_heartbeat_timeout
        )
        self.global_heartbeat  # prevent unused variable warning

    def node_status_pub(self):
        self.time = self.get_clock().now()

        heartbeat = Heartbeat()

        # NOTE: udpating the node status to be in line with the global status
        if (self.global_status != ERROR_STATE and self.node_status != ERROR_STATE): 
            node_status = self.global_status
        else: # can add more logic here
            node_status = ERROR_STATE

        if node_status != self.node_status: 
            self.get_logger().info(f"publishing {self.node_name} status: {self.node_status}")

        self.node_status = node_status

        heartbeat.status = self.node_status
        heartbeat.header.stamp = self.time.to_msg()
        heartbeat.header.frame_id = self.node_name

        self.status_publisher.publish(heartbeat)

    def check_global_heartbeat_timeout(self):
        if self.get_clock().now() - self.last_global_heartbeat > self.threshold:
            self.global_status = ERROR_STATE
            self.get_logger().info(f'GLOBAL HEARTBEAT TIMEOUT!')

    def update_status(self, msg):
        self.global_status = msg.status
        self.last_global_heartbeat = self.get_clock().now()
        # self.get_logger().info(f"receving hb status: {msg.status}")

    def panic(self):
        self.node_status = ERROR_STATE
    
    def clear_error(self):
        self.node_status = ALIVE_STATE

    def alive(self):
        return (self.node_status != ERROR_STATE)


class Perceptions(HeartbeatNode):
    def __init__(self):
        super().__init__('perceptions')
        self.count = 0  
        self.timer = self.create_timer(0.5, self.increment_count)

    def increment_count(self):
        self.count += 1
        print(self.count)
        if not self.alive(): 
            print("ERROR")
        if self.count > 10: 
            self.panic()

class Planning(HeartbeatNode):
    def __init__(self):
        super().__init__('planning')
        self.count = 0  
        self.timer = self.create_timer(1, self.increment_count)

    def increment_count(self):
        self.count += 1
        if not self.alive(): 
            print("ERROR")
        if self.count > 100: 
            self.panic()

class DIM(HeartbeatNode):
    def __init__(self):
        super().__init__('dim')
        self.count = 0  
        self.timer = self.create_timer(1, self.increment_count)
        self.state_subscriber = self.create_subscription(
                String,
                "DIM_request",
                self.process_state_request,
                10
        )

    def process_state_request(self, msg):
        self.get_logger().info(f"DIM REQUEST: {msg.data}")

    def increment_count(self):
        self.count += 1
        if not self.alive(): 
            print("ERROR")

def dim(args=None):
    rclpy.init(args=args)

    dim = DIM()

    rclpy.spin(dim)

    dim.destroy_node()
    rclpy.shutdown()

def perceptions(args=None):
    rclpy.init(args=args)

    perceptions = Perceptions()

    rclpy.spin(perceptions)

    perceptions.destroy_node()
    rclpy.shutdown()

def planning(args=None):
    rclpy.init(args=args)

    perceptions = Planning()

    rclpy.spin(perceptions)

    perceptions.destroy_node()
    rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    node = Heartbeat("node")

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
