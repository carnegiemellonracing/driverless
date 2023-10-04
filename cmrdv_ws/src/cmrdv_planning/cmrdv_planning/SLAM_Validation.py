from xmlrpc.server import DocXMLRPCRequestHandler
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Int8
from nav_msgs.msg import Odometry #TODO:make sure that the driver is actually installed
# from sbg_driver.msg import SbgGpsPos #TODO: make sure that the sbg drivers properly installed
import message_filters #TODO Make sure that this is installed too
from cmrdv_interfaces.msg import VehicleState, ConePositions,ConeList #be more specific later if this becomes huge
from cmrdv_common.config.collection_config import BEST_EFFORT_QOS_PROFILE
from cmrdv_common.config.planning_config import *
# from cmrdv_ws.src.cmrdv_planning.planning_codebase.ekf.map import *
from cmrdv_ws.src.cmrdv_planning.planning_codebase.ekf.new_slam import *
# from cmrdv_ws.src.cmrdv_planning.planning_codebase.graph_slam import *
from cmrdv_interfaces.msg import *
import numpy as np
import math
import time
from transforms3d.quaternions import axangle2quat

#import all of the sufs messages
from eufs_msgs.msg import *


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
        self.subscription_cone_data = message_filters.Subscriber(self, ConeArrayWithCovariance, '/cones')
        # self.subscription_cone_data.registerCallback(self.parse_cones)

        #Subscribe to Vehichle State
        self.subscription_vehicle_data = message_filters.Subscriber(self, CarState, '/ground_truth/state')
        # self.subscription_vehicle_data.registerCallback(self.parse_state)

        #Synchronize gps and odometry data
        self.ts = message_filters.ApproximateTimeSynchronizer([self.subscription_cone_data, self.subscription_vehicle_data], 10, slop=0.05)
        self.ts.registerCallback(self.runSLAM)

        self.subscription_cone_data  # prevent unused variable warning
        self.subscription_vehicle_data  # prevent unused variable warning

        # #Publishing Vehicle state and map of track
        # self.publisher_vehicle_state = self.create_publisher(VehicleState, VEHICLE_STATE_TOPIC, qos_profile=BEST_EFFORT_QOS_PROFILE)
        # self.publisher_cone_positions = self.create_publisher(ConePositions, MAP_TOPIC, qos_profile=BEST_EFFORT_QOS_PROFILE)
        # self.publisher_lap_num = self.create_publisher(Int8, LAP_NUM_TOPIC, qos_profile=BEST_EFFORT_QOS_PROFILE)

    
        #SLAM Variables
        self.ekf_output = None
        self.optimized = False
        self.gs_vehicle_state = None
        self.ekf_vehicle_state = None
        self.cone_positions = None

        # self.EKF = Map()
        # self.GraphSlam = GraphSlam(lc_limit=5)


        self.cones = None
        #robot state
        self.state = None

        self.xEst = np.zeros((STATE_SIZE, 1))
        self.pEst = np.eye(STATE_SIZE)
        #Time  Variables
        self.prevTime = self.get_clock().now().to_msg()
        self.dTime = 0
        self.car_states = []
        self.gt_states = []
        self.all_cones = []

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
        # return parsed_cones
    
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
        q0, q1, q2, q3 = quat.w, quat.x, quat.y, quat.z
        # following 11c from here: https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
        yaw_heading = math.atan2(2*(q0*q3 + q1*q2), q0**2+q1**2-q2**2-q3**2)
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
        #shouldn't be yaw_quat, should be yaw_angle i.e. heading
        #TODO need to figure out how to convert the quat to yaw_angle
        self.state = np.array([x,y,yaw_heading,dx,dy,dyaw,ddx,ddy,ddyaw])
        return self.state


    def runSLAM(self, cones_msg, state_msg):
        self.parse_cones(cones_msg)
        self.parse_state(state_msg)
        self.runEKF(self.get_clock().now().to_msg())
        # self.get_logger().info(f'{self.ekf_output}')

        # self.gs_vehicle_state, self.cone_positions, self.optimized = self.runGraph(self.ekf_output)
        # if self.optimized == True:
            # p_msg = self.vehicleStateToMsg(self.gs_vehicle_state)
            # self.publisher_vehicle_state.publish(p_msg)
            # self.get_logger().info('Publishing GraphSlam Vehicle State:')

            # p_msg = self.conePositionsToMsg(self.cone_positions)
            # self.publisher_cone_positions.publish(p_msg)
            # self.get_logger().info('Publishing GraphSlam Cone Positions:')
        # else:
        #     p_msg = self.vehicleStateToMsg(self.ekf_vehicle_state)
        #     self.publisher_vehicle_state.publish(p_msg)
        #     self.get_logger().info('Publishing EKF Vehicle State:')
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
    
    def updateTime(self,time): #Time data structure from header
        # self.get_logger().info(f'{type(self.prevTime)} {type(time)}')
        curr_time_ns = time.sec*1e9 + time.nanosec
        prev_time_ns = self.prevTime.sec*1e9 + self.prevTime.nanosec
        self.dTime = (curr_time_ns-prev_time_ns)/1e9
        self.get_logger().info(f'Current Time: {time.nanosec}')
        self.get_logger().info(f'Prev Time: {self.prevTime.nanosec}')
        # self.get_logger().info(f"time.sec: {time.sec}")
        self.prevTime = time

    def pi_2_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    # def print_xEst(self):
    #     # self.get_logger().info(f'Robot Pose: {self.xEst[0, 0], self.xEst[1, 0], self.xEst[2, 0] }')
    #     # self.get_logger().info(f'{(self.xEst.shape[0]-3)/2}')
    #     for i in range(int((self.xEst.shape[0]-3)/2)):
    #         # self.get_logger().info(f'   Landmark {i}: {self.xEst[2*i+3, 0]}, {self.xEst[2*i+4, 0]}')
    #     # self.get_logger().info(f'------------------------------------------')

    def runEKF(self, time_stamp): #TODO:verify message of this
        self.updateTime(time_stamp)
        car_x, car_y, car_heading = self.state[0], self.state[1], self.state[2]
        u = np.array([[math.hypot(self.state[3], self.state[4]), self.state[5]]]).T
        self.get_logger().info(f"linear: {u[0, 0]} | angular: {u[1, 0]}")
        z = np.zeros((0, 3))
        i = 0
        for x, y, _ in self.cones:
            dx = x
            dy = y
            dist = math.hypot(dx, dy)
            angle = self.pi_2_pi(math.atan2(dy, dx))
            zi = np.array([dist, angle, i])
            z = np.vstack((z, zi))
        self.xEst, self.pEst, cones = ekf_slam(self.xEst, self.pEst, u, z, self.dTime, self.get_logger())
        self.all_cones.extend(cones)
        self.get_logger().info(f'Num Landmarks = {(self.xEst.shape[0]-3)/2}')
        # self.print_xEst()
        self.plot_state_matrix()
        #TODO: Add dt
        # somelist = self.EKF.robot_cone_state()
        # self.get_logger().info(f'map = {map}')
        
        # self.plot_state_matrix(map, n_landmarks)

    def plot_state_matrix(self):
        plt.cla()
        plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        self.car_states.append(self.xEst[0:3])
        self.gt_states.append([self.state[0], self.state[1], self.state[2]])
        self.get_logger().info(f'Ground Truth: {self.state[0]}, {self.state[1]}, {self.state[2]}')
        self.get_logger().info(f'Final Output: {self.xEst[0, 0]}, {self.xEst[1, 0]}, {self.xEst[2, 0]}')
        # plt.plot(self.xEst[0, 0], self.xEst[1, 0], "or")
        for gt_state in self.gt_states:
            x, y, theta = gt_state[0], gt_state[1], gt_state[2]
            r = 1
            plt.arrow(x, y, r*math.cos(theta), r*math.sin(theta), color='red')
        for car_state in self.car_states:
            x, y, theta = car_state[0, 0], car_state[1, 0], car_state[2, 0]
            r = 1
            # self.get_logger().info(f'{x}, {y}, {theta}, {r*math.cos(theta)}, {r*math.sin(theta)}')
            plt.arrow(x, y, r*math.cos(theta), r*math.sin(theta))
        # for cone in self.all_cones:
        #     x, y = cone[0], cone[1]
        #     plt.plot(x, y, "or")
        for i in range(int((self.xEst.shape[0]-3)/2)):
            plt.plot(self.xEst[STATE_SIZE + i * 2],
                     self.xEst[STATE_SIZE + i * 2 + 1], "xg")
        # for x, y, _ in self.cones:
        #     plt.plot(x, y, "or")
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)
        # m_to_pix_factor_y = 35
        # m_to_pix_factor_x = 50
        # img = np.zeros([1024,1024,1],dtype=np.uint8)
        # img.fill(255)
        # robot_pose = (500, 900)
        # img = cv2.circle(img, (np.round(robot_pose[0]).astype("int"), np.round(robot_pose[1]).astype("int")), radius=1, color=(0, 0, 255), thickness=2)
        # for i in range(n_landmarks):
        #     idx = 2*i + 3
        #     pix_x = somelist[idx]
        #     pix_y = somelist[idx+1]
        #     cone_x = robot_pose[0] + pix_x*m_to_pix_factor_x
        #     cone_y = robot_pose[1] - pix_y*m_to_pix_factor_y
        #     # self.get_logger().info(f'x = {pix_x} -> {cone_x}     |  y = {pix_y} -> {cone_y}')
        #     img = cv2.circle(img, (int(cone_x), int(cone_y)), radius=1, color=(0, 255, 0), thickness=2)
        # cv2.imshow('current_map', img)
        # cv2.waitKey(10)

    # def runGraph(self, ekf_output):
    #     self.get_logger().info('EKF output recieved:')
    #     self.get_logger().info('Running GraphSlam')
    #     return self.GraphSlam.run_graph_slam(ekf_output)

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
