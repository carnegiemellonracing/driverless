import rclpy
from rclpy.node import Node
from rclpy.time import Time

from cmrdv_interfaces.msg import Heartbeat 
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.collection_config import RELIABLE_QOS_PROFILE
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.common_config import HEARTBEAT_REQUIRED_NODES
from cmrdv_ws.src.cmrdv_common.cmrdv_common.CAN.can_types import ALIVE_STATE, WARNING_STATE, ERROR_STATE, TX10HZ_PERIOD_S

# from cmrdv_common.config.collection_config import RELIABLE_QOS_PROFILE
# from cmrdv_common.config.common_config import HEARTBEAT_REQUIRED_NODES
# from cmrdv_common.CAN.can_types import ALIVE_STATE, WARNING_STATE, ERROR_STATE, TX10HZ_PERIOD_S, STATE

class NodeStatus():
    """
    Simple struct-like class to store status of each node.
    """
    def __init__(self, name) -> None:
        self.name = name
        self.last_time = 0
        self.status = ALIVE_STATE
        self.started = False
        self.subscription = None

    def __str__(self):
        return str(self.__dict__)

class GlobalHeartbeat(Node):
    """
    Global Heartbeat object publishes its current status at 10Hz and processes the vital heartbeats at 10Hz. 
    Publishes its current state to /global_heartbeat type: cmrdv_interfaces/Heartbeat
    # TODO: determine what other nodes need to be included in HEARTBEAT_REQUIRED_NODES

    Parameters
    ----------
    timer_period : float
        Frequency to publish status to system
    """
    def __init__(self, timer_period=TX10HZ_PERIOD_S):
        super().__init__('global_heartbeat')
        # Should we have it beating at the beginning? 
        self.beating = True

        self.global_heartbeat = self.create_publisher(
            Heartbeat,
            'global_heartbeat',
            RELIABLE_QOS_PROFILE
        )

        self.timer = self.create_timer(timer_period, self.hb_handler)
        # NOTE: this can be changed but this should be about 3-4 timer periods, currently 4 timer periods
        self.threshold = rclpy.time.Duration(seconds=(4 * timer_period))

        # create a node for each vital node
        self.nodes = {}
        self.update_required_nodes()

    def update_required_nodes(self):
        for node in HEARTBEAT_REQUIRED_NODES:
            if node in self.nodes:
                continue
            self.nodes[node] = NodeStatus(node)
            self.nodes[node].subscription = self.create_subscription(Heartbeat, f"{node}_status", self.node_check, RELIABLE_QOS_PROFILE)
            self.nodes[node].last_time = self.get_clock().now()

    def node_check(self, msg):
        """
        Checks each node's header time stamp against its previous header time stamp. 
        If the state of the msg is ERROR or the time stamp is greater than a threshold, then ERROR state, 
        otherwise, ALIVE state.
        """
        node = self.nodes[msg.header.frame_id] 

        if not node.started:
            node.started = True
            node.last_time = self.get_clock().now()

        # create Time object from the header
        curr_time = Time.from_msg(msg.header.stamp)

        # checking node status          
        # checking last time node published message
        if msg.status != ERROR_STATE and (curr_time - node.last_time <= self.threshold):
            node.status = ALIVE_STATE
        # NOTE: can add more cases for warnings etc. 
        else:
            node.status = ERROR_STATE
        node.last_time = curr_time
        # for debugging 
        self.get_logger().info(f"{msg.header.frame_id}: {msg.status} time: {curr_time - node.last_time}")


    def hb_handler(self):
        """
        For each of the node heartbeats, check if the status is alive, 
        if it is still publishing properly, and if node is started.
        """
        curr_time = self.get_clock().now()

        for node_name in self.nodes:
            node = self.nodes[node_name]
            node_ok = ((node.status != ERROR_STATE and                       # node is not erroring
                        (curr_time - node.last_time <= self.threshold)) or   # node is still publishing
                        (not node.started))                                  # node has started

            self.get_logger().info(f"{node_name} received: ERROR_STATE: {node.status != ERROR_STATE}, \
                                    TIMEOUT: {curr_time - node.last_time <= self.threshold}", once=True)
            self.beating = self.beating and node_ok

        heartbeat = Heartbeat()
        if self.beating:
            heartbeat.status = ALIVE_STATE
        # NOTE: again can add more logic for warnings etc. 
        else: 
            heartbeat.status = ERROR_STATE 

        heartbeat.header.frame_id = "global_heartbeat"
        heartbeat.header.stamp = curr_time.to_msg()

        # self.get_logger().info(f"publishing global heatbeat: {heartbeat.status}")  
        self.global_heartbeat.publish(heartbeat)

    def panic(self):
        """
        Immediately tell the system to error out by setting the alive boolean to False
        """
        self.beating = False

    def clear_error(self):
        """
        Tell the system to reset itself by setting the alive boolean to True
        """
        self.beating = True

    def alive(self):
        """
        Whether or not the system is et to error out
        """
        return self.beating


def main(args=None):
    rclpy.init(args=args)

    heartbeat = GlobalHeartbeat()

    rclpy.spin(heartbeat)

    heartbeat.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

