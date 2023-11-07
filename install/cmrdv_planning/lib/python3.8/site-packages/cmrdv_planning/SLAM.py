from xmlrpc.server import DocXMLRPCRequestHandler
import rclpy
from rclpy.node import Node

from std_msgs.msg import Int8
from nav_msgs.msg import Odometry #TODO:make sure that the driver is actually installed
from sbg_driver.msg import SbgGpsPos #TODO: make sure that the sbg drivers properly installed
import message_filters #TODO Make sure that this is installed too
from cmrdv_interfaces.msg import VehicleState, ConePositions,ConeList #be more specific later if this becomes huge
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.collection_config import BEST_EFFORT_QOS_PROFILE
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.planning_config import *
from cmrdv_ws.src.cmrdv_planning.planning_codebase.ekf.map import *
# from cmrdv_ws.src.cmrdv_planning.planning_codebase.graph_slam import *
from cmrdv_interfaces.msg import *
import numpy as np
from transforms3d import axangle2quat



#import odometry and gps position from sbg sensor driver
class SLAMSubscriber(Node):

    def __init__(self):
        super().__init__('slam_subscriber')
        # self.subscription_cone_data = self.create_subscription(
        #     np.array, #TBD by perceptions
        #     CONE_DATA_TOPIC
        #     'topic', #TBD by peerceptions
        #     self.runSLAM,
        #     10)
        # self.subscription_sensor_data = self.create_subscription(
        #     np.array, #TBD by software
        #     SENSOR_DATA_TOPIC
        #     'topic', #TBD by software
        #     self.runSLAM,
        #     10)

        #Subscribe to Perceptions Data #this is wrong - maybe need ,self.parsePerceptionCones
        self.subscription_cone_data = message_filters.Subscriber(CONE_DATA_TOPIC, ConeList)


        #Subscribe to Sensor data from SBG Sensor
        #self.subscription_sensor_data = message_filters.Subscriber(SENSOR_DATA_TOPIC, TBD)
        self.subscription_odom_data = message_filters.Subscriber(IMU_ODOM_TOPIC, Odometry,self.parse_imu)
        self.subscription_gps_data = message_filters.Subscriber(SBG_GPS_TOPIC, SbgGpsPos,self.parse_sbg)
        #Synchronize gps and odometry data
        self.ts = message_filters.TimeSynchronizer([self.subscription_cone_data, self.subscription_sensor_data], 10)
        self.ts.registerCallback(self.runSLAM)

        self.subscription_cone_data  # prevent unused variable warning
        self.subscription_gps_data  # prevent unused variable warning

        #Publishing Vehicle state and map of track
        self.publisher_vehicle_state = self.create_publisher( VehicleState,VEHICLE_STATE_TOPIC,qos_profile=BEST_EFFORT_QOS_PROFILE)
        self.publisher_cone_positions = self.create_publisher( ConePositions,MAP_TOPIC,qos_profile=BEST_EFFORT_QOS_PROFILE)

        
        self.publisher_lap_num = self.create_publisher(Int8,LAP_NUM_TOPIC,qos_profile=BEST_EFFORT_QOS_PROFILE)

    
        self.ekf_output = None
        self.optimized = False
        self.gs_vehicle_state = None
        self.ekf_vehicle_state = None
        self.cone_positions = None

        #robot state
        self.x = None
        self.y = None
        self.yaw_quat = None
        self.dx= None
        self.dy = None
        self.dyaw = None
        self.ddx = None
        self.ddy = None
        self.ddyaw = None

        #Time  Variables
        self.prevTime = None
        self.dTime = None
 
        self.EKF = Map()
        # self.GraphSlam = GraphSlam()


    def parse_sbg(self,msg):
        #change longitude and latitude to x and y
        #we need a reference position when initalizing
        self.long = msg.longitude
        self.lat = msg.latitude
    #SLAM functions

    # def runSLAM(self,msg):
    #     self.ekf_output = self.runEKF(msg)
    #     self.gs_vehicle_state, self.cone_positions, self.optimized = self.runGraph(self.EKF_Output)
    #     if self.optimized == True:

    #         msg = self.vehicleStateToMsg(self.gs_vehicle_state)
    #         self.publisher_vehicle_state.publish(msg)
    #         self.get_logger().info('Publishing GraphSlam Vehicle State:')

    #         msg = self.conePositionsToMsg(self.cone_positions)
    #         self.publisher_cone_positions.publish(msg)
    #         self.get_logger().info('Publishing GraphSlam Cone Positions:')
    #     else:
    #         msg = self.vehicleStateToMsg(self.ekf_vehicle_state)
    #         self.publisher_vehicle_state.publish(msg)
    #         self.get_logger().info('Publishing EKF Vehicle State:')

    def runSLAM(self,s_msg):
        self.ekf_output = self.runEKF(s_msg)
        # self.gs_vehicle_state, self.cone_positions, self.optimized = self.runGraph(self.ekf_output)
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
        
        # if(self.GraphSlam.lap > 1):
        #     msg = Int8()
        #     msg.data = self.GraphSlam.lap
        #     self.publisher_lap.publish(msg)

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
    
    def updateTime(self,time): #TODO: time data structure from header
        self.dTime = self.prevTime-time
        self.prevTime = time

    
    def runEKF(self, msg):
        #Parse perceptions data:
        perceptionsData = msg[0].data #TBD
        parsedConeData= self.parsePerceptionCones(perceptionsData)
        self.get_logger().info('Cone data recieved:')

        #Parse Sensor Data
        sensorData = msg[1].data #TBD
        state = self.parse_imu(sensorData)
        self.get_logger().info('Sensor data recieved:')

        #Update time and difference
        self.updateTime(msg[1].data.header.stamp)


        #TODO: fill out with actual implementation
        self.EKF.update_map(parsedConeData)

        #TODO: figure out how to get dt (sensor data???) ->add this in
        if self.EKF.updated_cone:
            somelist = self.EKF.robot_cone_state() #change variable name
        return somelist

        
    def parsePerceptionCones(coneData):
        list = coneData.blue_cones
        arr = []
        for elem in list:
            lis = [elem.x,elem.y,0]
            arr.append(lis)
        
        list = coneData.yellow_cones
        for elem in list:
            lis = [elem.x,elem.y,1]
            arr.append(lis)
        
        list = coneData.orange_cones
        for elem in list:
            lis = [elem.x,elem.y,2]
            arr.append(lis)
        
        arr = np.array(arr)
        return arr
    
    def parse_imu(self,msg):
        #we need x,y,dx,dy,yaw,velocity of yaw
        twist = msg.twist.twist
        pose = msg.pose.pose.position
        acceleration = msg.acceleration
        
        #x,y pos
        self.x = pose.x
        self.y = pose.y
        #yaw
        quat = msg.pose.pose.orientation
        self.yaw_quat = quat

        #lin velocity x and y
        self.dx = twist.linear.x
        self.dy = twist.linear.y

        #ang velocity-> already in yaw
        self.dyaw = twist.angular.z

        #acceleration
        self.ddx = acceleration.linear.x
        self.ddy = acceleration.linear.y
        self.ddyaw = acceleration.angular.z #angular acceleration


        state = np.array([self.x,self.y,self.yaw_quat,self.dx,self.dy,self.dyaw,self.ddx,self.ddy,self.ddyaw])
        return state


    def runGraph(self, ekf_output):
        self.get_logger().info('EKF output recieved:')
        self.get_logger().info('Running GraphSlam')
        # return self.GraphSlam.run_graph_slam(ekf_output)

def main(args=None):
    print("Initializing SLAM")
    rclpy.init(args=args)

    slam_subscriber = SLAMSubscriber()
    slam_subscriber.get_logger().info('Currently running')
    rclpy.spin(slam_subscriber)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    slam_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
