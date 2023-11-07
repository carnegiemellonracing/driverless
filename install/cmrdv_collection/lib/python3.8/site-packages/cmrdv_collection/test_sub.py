import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2
from cmrdv_common.cmrdv_common.config import collection_config as cfg
from eufs_msgs.msg import ConeArrayWithCovariance
from cmrdv_interfaces.msg import SimDataFrame

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(SimDataFrame, '/data_time_sync', self.listener_callback, 50)
        self.publisher = self.create_publisher(PointCloud2, '/urmom_pts', 30)
        self.subscription  # prevent unused variable warning
        self.i = 0

    def listener_callback(self, msg):
        print(f'got msg from /data_time_sync and publishing pointcloud2 to /urmom_pts - {self.i}')
        self.publisher.publish(msg.zed_pts)
        self.i = self.i + 1 

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
