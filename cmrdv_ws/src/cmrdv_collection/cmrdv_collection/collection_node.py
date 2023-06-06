import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from message_filters import ApproximateTimeSynchronizer, Subscriber, Cache
from sensor_msgs.msg import Image, PointCloud2
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config import collection_config as cfg
from cmrdv_interfaces.msg import DataFrame
import pdb
from sbg_driver.msg import SbgGpsPos, SbgImuData
import time

class CollectionNode(Node):
    def __init__(self):
        super().__init__("CollectionNode")

        collection_time = 0.1  # in secs

        # self.zed_left_sub = Subscriber(self, Image, cfg.ZED_LEFT, qos_profile=qos_profile_sensor_data)
#        self.vlp16_sub = Subscriber(self, PointCloud2, cfg.VLP16, #qos_profile=qos_profile_sensor_data)
        # self.zed_pts_sub = Subscriber(self, PointCloud2, cfg.ZED_POINTS, qos_profile=qos_profile_sensor_data)
        self.sbg_sub = Subscriber(self, SbgGpsPos, cfg.SBG, qos_profile=qos_profile_sensor_data) #TODO: Update msg type
        self.imu_sub = Subscriber(self, SbgImuData, cfg.IMU, qos_profile=qos_profile_sensor_data) #TODO: Update msg type
       
        self.use_sbg_imu = False
        
        self.caches = [Cache(self.zed_left_sub, cache_size=100), 
 #                      Cache(self.vlp16_sub, cache_size=100), 
                       Cache(self.zed_pts_sub, cache_size=100)
                       ]

        # NOTE: adding sbg and imu sensors here
        # this is temporary until using the sbg and imu collection starts working
        if self.use_sbg_imu:
            self.caches.append(Cache(self.sbg_sub, cache_size=10))
            self.caches.append(Cache(self.imu_sub, cache_size=10))

        #TODO: Verify publisher message type declaration
        self.data_pub = self.create_publisher(DataFrame, cfg.DATA_TIME_SYNC, cfg.QUEUE_SIZE)

        
        self.data_syncer = self.create_timer(collection_time, self.sync)

        self.caches[0].registerCallback(self.zed_left_callback)
        #self.caches[1].registerCallback(self.vlp16_callback)
        self.caches[1].registerCallback(self.zed_pts_callback)

        if self.use_sbg_imu:
            self.caches[3].registerCallback(self.sbg_callback)
            self.caches[4].registerCallback(self.imu_callback)

    def zed_left_callback(self, message):
        # self.caches[0].add(message)
        pass

    def vlp16_callback(self, message):
        # self.caches[1].add(message)
        pass

    def zed_pts_callback(self, message):
        # self.caches[2].add(message)
        pass

    def sbg_callback(self, message):
        # self.caches[3].add(message)
        pass
    def imu_callback(self, message):
        # self.caches[4].add(message)
        pass
    
    def sync(self):
        # pdb.set_trace()
        start = time.time()
        oldest_time = None
        for i in range(len(self.caches)):
            ts = self.caches[i].getLastestTime()
            if ts is None:
                print("Data collection cache", i, "is empty. Skipping message.")
                return

            #TODO: Add safety for when some cache has no elements

            if oldest_time is None or ts < oldest_time:
                oldest_time = ts

        best = []

        for i in range(len(self.caches)):
            ts_next = self.caches[i].getElemAfterTime(oldest_time)
            if ts_next is not None:
                ts_next_time = ts_next.header.stamp.sec * 10**9 + ts_next.header.stamp.nanosec

            ts_prev = self.caches[i].getElemBeforeTime(oldest_time)
            if ts_prev is not None:
                ts_prev_time = ts_prev.header.stamp.sec * 10**9 + ts_prev.header.stamp.nanosec

            #TODO: Check what happens on empty cache after time or before time
            
            if ts_prev is None or (ts_next is not None and ts_next_time - oldest_time.nanoseconds < oldest_time.nanoseconds - ts_prev_time):
                best.append(ts_next)
            else:
                best.append(ts_prev)


        msg = DataFrame()

        if self.use_sbg_imu:
            msg.zed_left_img, msg.vlp16_pts, msg.zed_pts, msg.sbg, msg.imu = best
        else:
            msg.zed_left_img, msg.zed_pts = best

        print(f"elapsed time for sync: {time.time() - start}")
        self.data_pub.publish(msg)

    
def main(args=None):
    rclpy.init(args=args)

    collection = CollectionNode()

    rclpy.spin(collection)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    collection.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
