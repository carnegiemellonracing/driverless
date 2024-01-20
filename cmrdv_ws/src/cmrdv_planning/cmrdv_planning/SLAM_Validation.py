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

AVG_COMPUTATION_TIME = 0 
WORST_COMPUTATION = () #first elem: computation time; second elem: idx to ground_truth cone
data_association_errors = 0
#import odometry and gps position from sbg sensor driver
class SLAMSubscriber(Node):

    def __init__(self):
        super().__init__('slam_subscriber')
        #Subscribe to Perceptions Data #this is wrong - maybe need ,self.parsePerceptionCones
        # self.subscription_cone_data = message_filters.Subscriber(CONE_DATA_TOPIC, ConeList)
        self.subscription_cone_data = message_filters.Subscriber(self, ConeArrayWithCovariance, '/ground_truth/cones')
        # self.subscription_cone_data.registerCallback(self.parse_cones)

        #Subscribe to Vehichle State
        self.subscription_vehicle_data = message_filters.Subscriber(self, CarState, '/ground_truth/state')
        # self.subscription_vehicle_data.registerCallback(self.parse_state)

        #Synchronize gps and odometry data
        self.ts = message_filters.ApproximateTimeSynchronizer([self.subscription_cone_data, self.subscription_vehicle_data], 10, slop=0.05)
        self.ts.registerCallback(self.runSLAM)

        self.subscription_cone_data  # prevent unused variable warning
        self.subscription_vehicle_data  # prevent unused variable warning
        self.min_id_errors = 0

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

        # the ground truth cones
        self.cones = None
        #robot state
        self.state = None

        self.xEst = np.zeros((STATE_SIZE, 1))
        self.xTruth = np.zeros((STATE_SIZE, 1))
        self.pEst = np.eye(STATE_SIZE)

        self.xErr = np.zeros(( int (((int(self.xEst.shape[0]) - 3))/2), 1))

        #Time  Variables
        self.prevTime = self.get_clock().now().to_msg()
        self.dTime = 0
        self.car_states = []
        self.gt_states = []

        # the ekf calculated cones
        self.all_cones = []
        self.missed_cones = 0
        self.missed_states = 0
        self.mean_error = 0
        self.stdev_error = 0
        self.bad_states = 0 # heuristic variables
        self.closest_data = []

    def parse_cones(self,msg):
        if self.cones is not None:
            self.get_logger().info("Num cones: " + (str)(len(self.cones)))
        cones = msg.blue_cones
        parsed_cones = []

        for cone in cones:
            tmp = [cone.point.x,cone.point.y,0]
            # self.get_logger().info("Testing cone subs; x: {cone.point.x}, y: {cone.point.y}");
            self.get_logger().info(f"Blue cone: x: {cone.point.x}, \t y: {cone.point.y}");
            parsed_cones.append(tmp)
        
        cones = msg.yellow_cones
        for cone in cones:
            tmp = [cone.point.x,cone.point.y,1]
            self.get_logger().info(f"Yellow cone: x: {cone.point.x}, \t y: {cone.point.y}");
            parsed_cones.append(tmp)
        
        cones = msg.orange_cones
        for cone in cones:
            tmp = [cone.point.x,cone.point.y,2]
            self.get_logger().info(f"Orange cone: x: {cone.point.x}, \t y: {cone.point.y}");

            parsed_cones.append(tmp)
        
        parsed_cones = np.array(parsed_cones)
        self.cones = parsed_cones
        self.closest_data = np.empty(len(self.cones), dtype = object)
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

    def almostEqual(self, n1, n2):
        return (abs(n1 - n2) <= 0.0001)


    def append_xTruth(self):
        assert(len(self.xTruth) >= 3)
        assert(len(self.state) > 0)
        assert(len(self.cones) > 0)
        self.xTruth[0][0] = self.state[0]
        self.xTruth[1][0] = self.state[1]
        self.xTruth[2][0] = self.state[2]

        obs_cones_w_idx = []
        for idx, cone in enumerate(self.cones):
            obs_cone_global = np.zeros((2, 1))
            # obs_cone_global[0][0] = self.state[0] + cone[0] #global x pos of cone
            # obs_cone_global[1][0] = self.state[1] + cone[1] #global y pos of cone
            # dist = math.hypot(dx, dy)
            relative_distance = self.euclid_dist(self.state[0], self.state[1], self.state[0] + cone[0], self.state[1] + cone[1])
            #remember that cone[0] and cone[1] represents the change in x and y distance from the car to the cone
            # relative_distance = math.hypot(cone[0], cone[1])
            # angle = self.pi_2_pi(math.atan2(dy, dx))
            # phi = math.atan2(cone[0], cone[1])
            # phi = self.pi_2_pi(math.atan2(cone[1], cone[0]))
            phi = math.atan(cone[1]/ cone[0])
            obs_cone_global[0][0] = self.state[0] + relative_distance * math.cos(self.state[2] + phi) #global x pos of cone
            obs_cone_global[1][0] = self.state[1] + relative_distance * math.sin(self.state[2] + phi) #global y pos of cone

            new_cone = True

            #data association performed by looking at whether a cone alrdy exists at position
            # these series of steps represent what xEst should do
            l = len(self.xTruth)
            for i in range(3, l, 2):
                if (self.almostEqual(self.xTruth[i][0], obs_cone_global[0][0]) and
                    self.almostEqual(self.xTruth[i+1][0], obs_cone_global[1][0])):
                    new_cone = False
                    obs_cones_w_idx.append([(i-3)/2, False])
                    break
            if new_cone:
                obs_cones_w_idx.append([(l - 3) / 2, True])
                self.xTruth = np.vstack((self.xTruth, obs_cone_global))
            print("Length of self.cones: ", len(self.cones))
        return obs_cones_w_idx 

    def runSLAM(self, cones_msg, state_msg):
        self.get_logger().info("SLAM Callback!")
        self.parse_cones(cones_msg) # is this parsing the ground truth cones??
        # subscription_cone_data is a message filter (gets data from topic)
        # what happens to the ground truth data???
        # how can I access it

        # subscription_cone_data is a message filter (gets data from topic)
        # what happens to the ground truth data???
        # how can I access it
        self.parse_state(state_msg)
        obs_cones_w_idx = self.append_xTruth()
        self.runEKF(self.get_clock().now().to_msg(), obs_cones_w_idx)
        # self.model_accuracy()
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
        curr_time_ns = time.sec*1e9 + time.nanosec #this is getting the time in nanoseconds
        # time.sec only gets the time in seconds which is not precise enough
        prev_time_ns = self.prevTime.sec*1e9 + self.prevTime.nanosec
        self.dTime = (curr_time_ns-prev_time_ns)/1e9
        # self.get_logger().info(f'Current Time: {time.nanosec}')
        # self.get_logger().info(f'Prev Time: {self.prevTime.nanosec}')
        # self.get_logger().info(f"time.sec: {time.sec}")
        self.prevTime = time

    # keep the absolute value of the angle < math.pi/2
    def pi_2_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    # def print_xEst(self):
    #     # self.get_logger().info(f'Robot Pose: {self.xEst[0, 0], self.xEst[1, 0], self.xEst[2, 0] }')
    #     # self.get_logger().info(f'{(self.xEst.shape[0]-3)/2}')
    #     for i in range(int((self.xEst.shape[0]-3)/2)):
    #         # self.get_logger().info(f'   Landmark {i}: {self.xEst[2*i+3, 0]}, {self.xEst[2*i+4, 0]}')
    #     # self.get_logger().info(f'------------------------------------------')get_logger().info(logger.
    
    def runEKF(self, time_stamp, obs_cones_w_idx): #TODO:verify message of this
        #self.updateTime(time_stamp)
        start_time = time_stamp.sec * 1e9 + time_stamp.nanosec
         
        
        car_x, car_y, car_heading = self.state[0], self.state[1], self.state[2]
        u = np.array([[math.hypot(self.state[3], self.state[4]), self.state[5]]]).T
        self.get_logger().info(f"linear: {u[0, 0]} | angular: {u[1, 0]}") # u describes the motion
        z = np.zeros((0, 3))
        i = 0
        for x, y, _ in self.cones: # self.cones comes from perceptions
            # These dx and dy are the relative positions of the cones
            dx = x
            dy = y
            dist = math.hypot(dx, dy)
            angle = self.pi_2_pi(math.atan2(dy, dx))
            zi = np.array([dist, angle, i])
            z = np.vstack((z, zi))
        self.xEst, self.pEst, cones, new_mi_errors = ekf_slam(self.xEst, self.pEst, u, z, self.dTime, self.get_logger(), self.xTruth, obs_cones_w_idx)
        
        self.min_id_errors += new_mi_errors

        # all_cones stores the calculated cones
        # self.all_cones.extend(cones)
        self.get_logger().info(f'Num Landmarks = {(self.xEst.shape[0]-3)/2}')

        # print("Type of subscription cone data: ", type(self.subscription_cone_data))
        # for thing in self.subscription_cone_data:
        #     print("Ground truth data: ", thing)
        # self.print_xEst()
        self.get_logger().info(f"CarxPos: {self.xTruth[0][0]},  yPos: {self.xTruth[1][0]}")

        for i in range(3, len(self.xTruth), 2):
            self.get_logger().info(f"Cone no. {1 + ((i - 3) / 2)}: \t xPos: {self.xTruth[i][0]},  yPos: {self.xTruth[i+1][0]}")
        self.plot_state_matrix()
        #TODO: Add dt
        # somelist = self.EKF.robot_cone_state()
        # self.get_logger().info(f'map = {map}')
        
        # self.plot_state_matrix(map, n_landmarks)
        next_time = self.get_clock().now().to_msg()
        next_time = next_time.sec*1e9 + next_time.nanosec
        diff_time = next_time - start_time

        global WORST_COMPUTATION
        if len(WORST_COMPUTATION) == 0:
            WORST_COMPUTATION = (diff_time, (self.xEst.shape[0]-3)/2)
        else:
            worst_time, n_landmark = WORST_COMPUTATION
            if diff_time > worst_time:
                WORST_COMPUTATION = (diff_time, (self.xEst.shape[0]-3)/2)
            worst_time = max(worst_time, diff_time)
            self.get_logger().info(f'Worst Time: {worst_time}')

        self.get_logger().info(f'Start Time: {start_time}')
        self.get_logger().info(f'Next Time: {next_time}')
        self.get_logger().info(f'Difference: {diff_time}')
        
        




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

    def print_model_acc(self):
        self.get_logger().info("This is the mean error: ")
        #self.get_logger().info(self.mean_error)
        self.get_logger().info("Num missed cones: ")
        #self.get_logger().info(self.missed_cones)
   
    def model_accuracy(self):
        #Fill Up closest_data
        for cone_coord in self.all_cones:   
            self.replace(self.closest_data, cone_coord[0, 0], cone_coord[1, 0])
            
        #Find temp mean error
        for pos in self.closest_data:
            if pos is None:
                self.missed_cones += 1
            else:
                self.mean_error += pos[2]
        self.mean_error /= len(self.closest_data) - self.missed_cones
        
        #Find temp standard deviation of error
        for pos in self.closest_data:
            if pos is not None:   
                self.stdev_error += (pos[2] - self.mean_error) * (pos[2] - self.mean_error)
        self.stdev_error = math.sqrt(self.stdev_error / (len(self.closest_data) - self.missed_cones))
        
        #Find statistically significant errors
        for ind in range(0, len(self.closest_data)):
            if self.closest_data[ind] is not None:   
                if abs(self.closest_data[ind][2] - self.mean_error) > 2 * self.stdev_error:
                    self.bad_states += 1
                    self.closest_data[ind] = None
                    
        #Reset mean and standard deviation
        self.mean_error = 0
        self.stdev_error = 0
        #Find mean error w/o statistically significant errors
        for pos in self.closest_data:
            if pos is not None:
                self.mean_error += pos[2]
        if len(self.closest_data) - self.missed_cones - self.bad_states != 0:
            self.mean_error /= len(self.closest_data) - self.missed_cones - self.bad_states
        
        #Find standard deviation of error w/o statistically significant errors
        for pos in self.closest_data:
            if pos is not None:   
                self.stdev_error += (pos[2] - self.mean_error) * (pos[2] - self.mean_error)
        if len(self.closest_data) - self.missed_cones - self.bad_states != 0:
            self.stdev_error = math.sqrt(self.stdev_error / (len(self.closest_data) - self.missed_cones - self.bad_states))
         
        self.print_model_acc()
        
    #Checks if the closest_data entry at ind is empty of it the current error is less
    def better_or_empty(self, closest_data, error, ind):
        if closest_data[ind] is not None:
            return error < closest_data[ind][2]
        return True
            
    #Places the ekf state at x, y into closest_data
    #If there's already a cone at the index:
    # 1) compare the current error with the error of the cone already ther
    # 2a.) if the error is less, then replace and recursive call on the next cone
    # 2b.) if the error is greater, then go to the next lowest
    def replace(self, closest_data, x, y):
        min_error = -1
        min_ind = -1
        
        #Find the closest ground state that the ekf state can be placed in
        for ind in range(0, len(self.cones)):
            dist = self.euclid_dist(x, y, self.cones[ind, 0], self.cones[ind, 1])
            if (min_ind == -1 or dist < min_error) and self.better_or_empty(closest_data, dist, ind):
                #Update min
                min_error = dist
                min_ind = ind 
        
        #Check that the current ekf state can be put into closest_data         
        if min_ind != -1:
            if closest_data[min_ind] is None:
                #Replace contents of closest data
                closest_data[min_ind] = np.array([x, y, min_error, self.cones[min_ind, 0], self.cones[min_ind, 1]])
            else:
                #Store old contents
                temp_x = closest_data[min_ind][0]
                temp_y = closest_data[min_ind][1]
                #Replace contents of closest data
                closest_data[min_ind] = np.array([x, y, min_error, self.cones[min_ind, 0], self.cones[min_ind, 1]])
                #Run same algorithm on replaced contents
                self.replace(closest_data, temp_x, temp_y)
        else:
            self.missed_states += 1
        
    #Finds the euclidean distance between (x1, y1) and (x2, y2)
    def euclid_dist(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
    


def main(args=None):
    rclpy.init(args=args)
    print("Initialize ")
    slam_subscriber = SLAMSubscriber()

    rclpy.spin(slam_subscriber)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    slam_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
