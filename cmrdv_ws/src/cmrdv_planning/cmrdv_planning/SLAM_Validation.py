from xmlrpc.server import DocXMLRPCRequestHandler
import rclpy
from rclpy.node import Node

from std_msgs.msg import Int8
from nav_msgs.msg import Odometry #TODO:make sure that the driver is actually installed
# from sbg_driver.msg import SbgGpsPos #TODO: make sure that the sbg drivers properly installed
import message_filters #TODO Make sure that this is installed too
from cmrdv_interfaces.msg import VehicleState, ConePositions,ConeList #be more specific later if this becomes huge
from cmrdv_common.cmrdv_common.config.collection_config import BEST_EFFORT_QOS_PROFILE
from cmrdv_common.cmrdv_common.config.planning_config import *
from cmrdv_planning.planning_codebase.ekf.map import *
from cmrdv_planning.planning_codebase.graph_slam import *
from cmrdv_interfaces.msg import *
import numpy as np
from transforms3d import axangle2quat

#import all of the sufs messages
from eufs.eufs_msgs.msg import *


VEHICLE_STATE_TOPIC = "/ground_truth/state"
VEHICLE_STATE_MSG = CarState
CONE_TOPIC = "/cones"
CONE_MSG = ConeArrayWithCovariance

#import odometry and gps position from sbg sensor driver
class SLAMSubscriber(Node):

    def __init__(self):
        super().__init__('slam_subscriber')
        #Subscribe to Perceptions Data #this is wrong - maybe need ,self.parsePerceptionCones
        # self.subscription_cone_data = message_filters.Subscriber(CONE_DATA_TOPIC, ConeList)
        self.subscription_cone_data = message_filters.Subscriber(CONE_TOPIC, ConeArrayWithCovariance,self.parse_cones)

        #Subscribe to Vehichle State
        self.subscription_vehicle_data = message_filters.Subscriber(VEHICLE_STATE_TOPIC, VEHICLE_STATE_MSG,self.parse_state)
        #Synchronize gps and odometry data
        self.ts = message_filters.TimeSynchronizer([self.subscription_cone_data, self.subscription_vehicle_data], 10)
        self.ts.registerCallback(self.runSLAM)

        self.subscription_cone_data  # prevent unused variable warning
        self.subscription_vehicle_data  # prevent unused variable warning

        #Publishing Vehicle state and map of track
        self.publisher_vehicle_state = self.create_publisher(VehicleState,VEHICLE_STATE_TOPIC,qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.publisher_cone_positions = self.create_publisher(ConePositions,MAP_TOPIC,qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.publisher_lap_num = self.create_publisher(Int8,LAP_NUM_TOPIC,qos_profile=BEST_EFFORT_QOS_PROFILE)

    
        #SLAM Variables
        self.ekf_output = None
        self.optimized = False
        self.gs_vehicle_state = None
        self.ekf_vehicle_state = None
        self.cone_positions = None

        self.EKF = Map()
        self.GraphSlam = GraphSlam()


        self.cones = None
        #robot state
        self.state = None

        #Time  Variables
        self.prevTime = None
        self.dTime = None
 

    def parse_cones(self,msg):
        cones = msg.blue_cones
        parsed_cones = []
        for cone in cones:
            tmp = [cone.point.x,cone.point.y,0]
            parsed_cones.append(tmp)
        
        cones = msg.yellow_cones
        for cone in cones:
            tmp = [cone.point.x,cone.point.y,1]
            parsed_cones.append(tmp)
        
        cones = msg.orange_cones
        for cone in cones:
            tmp = [cone.point.x,cone.point.y,2]
            parsed_cones.append(tmp)
        
        parsed_cones = np.array(parsed_cones)
        self.cones = parsed_cones
        return parsed_cones
    
    def parse_state(self,msg):
        #update time

        pose = msg.pose.pose.position
        twist = msg.twist.twist
        acceleration = msg.linear_acceleration
        
        #x,y pos
        x = pose.x
        y = pose.y
        #yaw
        quat = msg.pose.pose.orientation
        yaw_quat = quat

        #lin velocity x and y
        dx = twist.linear.x
        dy = twist.linear.y

        #ang velocity-> already in yaw
        dyaw = twist.angular.z

        #acceleration
        ddx = acceleration.x
        ddy = acceleration.y
        ddyaw = 0 #Not given by vehicle state: acceleration.angular.z 

        self.state = np.array([x,y,yaw_quat,dx,dy,dyaw,ddx,ddy,ddyaw])
        return self.state


    def runSLAM(self,s_msg):
        self.ekf_output = self.runEKF(s_msg)
        self.gs_vehicle_state, self.cone_positions, self.optimized = self.runGraph(self.ekf_output)
        if self.optimized == True:
            p_msg = self.vehicleStateToMsg(self.gs_vehicle_state)
            self.publisher_vehicle_state.publish(p_msg)
            self.get_logger().info('Publishing GraphSlam Vehicle State:')

            p_msg = self.conePositionsToMsg(self.cone_positions)
            self.publisher_cone_positions.publish(p_msg)
            self.get_logger().info('Publishing GraphSlam Cone Positions:')
        else:
            p_msg = self.vehicleStateToMsg(self.ekf_vehicle_state)
            self.publisher_vehicle_state.publish(p_msg)
            self.get_logger().info('Publishing EKF Vehicle State:')
        #publish if data collection lap has finished
        
        if(self.GraphSlam.lap > 1):
            msg = Int8()
            msg.data = self.GraphSlam.lap
            self.publisher_lap.publish(msg)

    def vehicleStateToMsg(self,vehicle_state): #6x1 array
        msg = VehicleState()
        msg.position.position.x = vehicle_state[0]
        msg.position.position.y = vehicle_state[1]
        quaternion = axangle2quat([0, 0, 1], vehicle_state[2]) #taking the ekf yaw isntead of sbg
        msg.position.orientation = quaternion
        #msg.position.orientation = self.yaw_quat
        msg.twist.linear.x = vehicle_state[3]
        msg.twist.linear.y = vehicle_state[4]
        msg.twist.angular.z = vehicle_state[5]

        msg.acceleration.linear.x = self.ddx
        msg.acceleration.linear.y = self.ddy 
        msg.acceleration.angular.z =  self.ddyaw #angular acceleration
    
        return msg

    def conePositionsToMsg(cone_positions):
        msg = ConePositions()
        msg.cone_positions = cone_positions
        return msg
    
    def updateTime(self,time): #Time data structure from header
        self.dTime = self.prevTime-time
        self.prevTime = time

    
    def runEKF(self, msg): #TODO:verify message of this
        #Parse perceptions data:
        self.get_logger().info('Cone data recieved:')
        #Parse Sensor Data
        self.get_logger().info('State data recieved:')
        #Update time and difference
        self.updateTime(msg[1].data.header.stamp)
        self.EKF.update_map(self.cones)
        #TODO: Add dt
        if self.EKF.updated_cone:
            somelist = self.EKF.robot_cone_state()
        return somelist

    def runGraph(self, ekf_output):
        self.get_logger().info('EKF output recieved:')
        self.get_logger().info('Running GraphSlam')
        return self.GraphSlam.run_graph_slam(ekf_output)

def main(args=None):
    rclpy.init(args=args)

    slam_subscriber = SLAMSubscriber()

    rclpy.spin(slam_subscriber)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    slam_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
