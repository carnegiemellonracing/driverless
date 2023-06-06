import rclpy
from rclpy.node import Node

from std_msgs.msg import Int8

import numpy as np
from cmrdv_ws.src.cmrdv_perceptions.utils.utils import np2points
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.collection_config import BEST_EFFORT_QOS_PROFILE
from cmrdv_interfaces.msg import CarROT,PairROT,ConeList,Points # be more specific later if this becomes huge
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.planning_config import *
from cmrdv_ws.src.cmrdv_planning.planning_codebase.midpoint.generator import *
import cmrdv_ws.src.cmrdv_planning.planning_codebase.raceline.frenet as frenet
from cmrdv_ws.src.cmrdv_planning.planning_codebase.raceline.raceline import Spline
from cmrdv_ws.src.cmrdv_common.cmrdv_common.DIM.dim_heartbeat import HeartbeatNode
import math



class Midpoint(HeartbeatNode):

    def __init__(self):
        super().__init__('midpoint')
        self.subscription_cones = self.create_subscription(
            ConeList,
            CONE_DATA_TOPIC,
            self.create_midpoint,
           # 10,
            qos_profile=BEST_EFFORT_QOS_PROFILE)

        self.subscription_lap_num = self.create_subscription(
            Int8, #TBD by software
            LAP_NUM_TOPIC,
            self.update_lap,
            10)
        
        self.subscription_cones  # prevent unused variable warning
        self.subscription_lap_num  # prevent unused variable warning

        #Carrot needs pose and curvature 
        self.publisher_ = self.create_publisher( PairROT, PAIRROT_TOPIC, qos_profile=BEST_EFFORT_QOS_PROFILE) 
        self.spline_publisher_ = self.create_publisher(Points, SPLINE_TOPIC, qos_profile=BEST_EFFORT_QOS_PROFILE) #we publish whenever we get data, so we don't need to send on a periodic basis 
        self.generator_ = MidpointGenerator(interpolation_number=10) 

        self.LOOKAHEAD_NEAR = 2
        self.LOOKAHEAD_FAR = 3
        self.vis_lookaheads = np.arange(1, 12, 0.5)
        self.lap = 1

        self.vis_spline = True

    def update_lap(self,msg):
        self.lap = msg.data


    def carrot_data(self,spline,lookahead):
        point, _, _, x  = spline.along(lookahead, precision=20) # TODO: replace by controls lookahead later
        print("point", point)

        # if self.vis_spline:
        #     # send the spline points for debugging
        #     spline_points = []
        #     for look in self.vis_lookaheads:
        #         vis_point, _, _, _ = spline.along(look, precision=20)
        #         spline_points.append([float(vis_point[0,0]), float(vis_point[0,1]), 0])
        #     msg = self.spline_to_msg(spline_points)
        #     self.spline_publisher_.publish(msg)

        # Compute curvature along the spline and yaw for the car
        curvature = frenet.get_curvature(spline.first_der,spline.second_der,(float(x)))
        deriv = spline.first_der(x) # gradient at the point
        print("first deriv:",deriv)
        yaw = math.pi/2-math.atan2(deriv, 1)-math.acos(spline.Q[0,0]) # angle formed by tangent with the curve
        yaw *= -1 #CHECK WITH GRIFF REMOVE IF WRONG
        return point,curvature,yaw

    #Subscribing-calls this whenever we get data
    #have it so whenever you recieve data, you process it and publish within a single function        
    #Midpoint Functions
    def create_midpoint(self, msg):
        print('Cone data received:')

        if self.lap > 1:
            print('Midpoint Spline ignores data: first lap done')
            self.destroy_node() #destroy node once midpoint is not running
            return

        perceptionsData = msg #numpy array

        # Do nothing if not enough data (remove hard-coded colors)
        if (len(perceptionsData.blue_cones) == 0 or len(perceptionsData.yellow_cones) == 0) and len(perceptionsData.orange_cones) < 2:
            print(f"Warning -- path-planning got"
                  f"- {len(perceptionsData.blue_cones)} blue/color=1 cones"
                  f"- {len(perceptionsData.yellow_cones)} yellow/color=2 cones"
                  f"- {len(perceptionsData.orange_cones)} orange/color=3 cones")
            print("Cannot create midpoint from this")
            return

        cones = self.parse_perceptions_cones(perceptionsData) #returns Nx3 numpy array of x,y,color

        # Generate spline fitting the midpoint defined by cones
        spline: Spline = self.generator_.spline_from_cones(cones)

        nearCarrotData = self.carrot_data(spline,self.LOOKAHEAD_NEAR)
        farCarrotData = self.carrot_data(spline,self.LOOKAHEAD_FAR)


        #yaw = math.pi/2 - math.atan2(point[0,1],point[0,0])
        # Convert to message for CarRot topic
        # publishing the path planning data to controls
        msg = self.midpoint_to_msg(nearCarrotData,farCarrotData)
        self.publisher_.publish(msg)

        self.get_logger().info('Publishing Midpoint Data:')

    def midpoint_to_msg(self,nearData,farData):
        
        (point_near, curvature_near, yaw_near) = nearData
        (point_far, curvature_far, yaw_far) = farData

        near = CarROT()
        #this is what controls is currently using from the carrot message
        near.x = float(point_near[0,0])
        near.y = float(point_near[0,1])
        near.yaw = float(yaw_near)
        near.curvature = float(curvature_near)
        print("near",near.x, near.y)

        far = CarROT()
        #this is what controls is currently using from the carrot message
        far.x = float(point_far[0,0])
        far.y = float(point_far[0,1])
        far.yaw = float(yaw_far)
        far.curvature = float(curvature_far)
        print("far",far.x, far.y)

        msg = PairROT()
        msg.near = near
        msg.far = far

        return msg

    def spline_to_msg(self, points):
        msg = Points()
        msg.points = np2points(points)
        return msg
    
    def parse_perceptions_cones(self,coneData):
        list = coneData.blue_cones
        arr = []
        for elem in list:
            temp = [elem.x,elem.y,1]
            arr.append(temp)
        
        list = coneData.yellow_cones
        for elem in list:
            temp = [elem.x,elem.y,2]
            arr.append(temp)
        
        list = coneData.orange_cones
        for elem in list:
            temp = [elem.x,elem.y,3]
            arr.append(temp)
        
        arr = np.array(arr)
        return arr



def main(args=None):
    rclpy.init(args=args)

    midpoint = Midpoint()
    
    print("Starting Midpoint node")

    rclpy.spin(midpoint)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    midpoint.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
