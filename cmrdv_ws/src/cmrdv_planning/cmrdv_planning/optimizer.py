from collections import namedtuple
import rclpy
from rclpy.node import Node

from std_msgs.msg import String,Int8,Float64MultiArray
import numpy as np
from transforms3d.quaternions import axangle2quat

from cmrdv_interfaces.msg import CarROT, VehicleState, ConeList,ConePositions #be more specific later if this becomes huge

from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.planning_config import *
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.collection_config import BEST_EFFORT_QOS_PROFILE
from cmrdv_ws.src.cmrdv_planning.planning_codebase.raceline.frenet import *
from cmrdv_ws.src.cmrdv_planning.planning_codebase.raceline.path_optimization import *
from cmrdv_ws.src.cmrdv_planning.planning_codebase.raceline.raceline import *
from cmrdv_ws.src.cmrdv_planning.planning_codebase.raceline.utils import *

Car = namedtuple('Car', ['x', 'y', 'vx', 'vy', 'yaw'])

class LTO(Node):
    def __init__(self):
        super().__init__('LTO')
        # Controls LOOKAHEAD
        self.LOOKAHEAD = 0.5

        #synchronize vehicle state and cone position
        self.slam_subscription_vehicle = self.create_subscription(
            VehicleState,
            VEHICLE_STATE_TOPIC,
            self.update_vehicle_state,
            10
        )
        
        self.slam_subscription_cones = self.create_subscription(
            ConeList,
            CONE_DATA_TOPIC,
            self.optimize,
            10
        )

        self.subscription_lap_num = self.create_subscription(
            Int8, #TBD by software
            LAP_NUM_TOPIC,
            self.update_lap,
            10
        )


        self.slam_subscription_vehicle  # prevent unused variable warning
        self.slam_subscription_cones  # prevent unused variable warning

        #Optimized path -> might not be needed, just save internally if not neede by other subteams
        self.optimized_path_publisher = self.create_publisher(Float64MultiArray,OPTIMIZED_PATH_TOPIC,qos_profile=BEST_EFFORT_QOS_PROFILE)

        #CarROT output for controls
        self.carrot_publisher = self.create_publisher( CarROT,RACELINE_PATH_TOPIC,qos_profile=BEST_EFFORT_QOS_PROFILE)


        self.optimized_path = None
        # Previous progress of the car (default not 0 as we may not exactly start from the beginning of the track)
        self.prev_progress = None
        self.cumulative_lengths = None
        self.optimized_path_states = None
        self.vehicle_state = None

        self.states = None
        self.lap = 0

    def update_lap(self,msg):
        self.lap = msg.data

    #Subscriber calls this whenever we get data
    #have it so whenever you receive data, you process it and publish within a single function        
    #Midline Functions
    def optimize(self, msg):
        '''
        Compute new optimized path whenever cone positions (in the world frame) are received
        '''
        self.get_logger().info('Map data received:')

        coneData = msg.data

        #helper function to make the 1d list into 2d list
        npArrayConedata = np.array(coneData)
        coneData2dlist = np.reshape(npArrayConedata,(-1,2))

        # Get centerline in the world frame
        _, generator = generate_centerline_from_cones(coneData2dlist, interpolation_number=0) # TODO: change this to respect signature of function

        reference_path = generator.cumulated_splines
        cumulative_lengths = generator.cumulated_lengths

        # Generate optimal path using with centerline as reference path
        path_optimizer = PathOptimizer(reference_path, cumulative_lengths, delta_progress=16)
        solution, _ = path_optimizer.optimize()
        states = solution[:, :8] # (progress_steps, n, mu, vx, vy, r, delta, T)
        self.states = states
        controls = solution[:, 8:] # change in delta and T

        # Convert solution states to points in (x, y) coordinates
        points = states_to_points(states, reference_path, cumulative_lengths)

        # Generate splines for those points
        [optimized_path, cumulative_lengths] = raceline_gen(points)

        self.optimized_path = optimized_path
        self.cumulative_lengths = cumulative_lengths

        #Figure out the actual parsing to get to the correct states
        #don't publish the actual race line
        #msg.path = states
        #self.publisher_.publish(msg)
        #self.get_logger().info('Publishing Optimized Path:')
        self.optimized_path_states = states
        #publishing the path planning data to controls
        self.optimized_path_publisher.publish(msg)

    def update_vehicle_state(self, msg):
        '''
        Update vehicle state with most recent information received from SLAM
        '''
        self.vehicle_state = Car(
            x=msg.position.x,
            y=msg.position.y,
            vx=msg.twist.linear.x,
            vy=msg.twist.linear.y,
            yaw=msg.twist.angular.z
        )
        self.publish_raceline_path()


    def publish_raceline_path(self):
        self.get_logger().info('Publishing Raceline Path:')

        if self.vehicle_state == None:
            self.get_logger().info('No information about the car, cannot predict next state')
            # TODO: Send an error to controls?
            return
        
        if self.lap <= 1:
            self.get_logger().info('Optimizer ignores data:')
            return

        # Project vehichle state on optimized path
        projection: Projection = frenet(self.vehicle_state.x, self.vehicle_state.y, self.optimized_path, 
                                        self.cumulative_lengths, self.prev_progress, self.vehicle_state.vx, 
                                        self.vehicle_state.vy)

        # Progress of first point on spline (will be 0 in the frame of the spline)
        spline_progress = self.cumulative_lengths[projection.min_index-1] if projection.min_index > 0 else 0

        # Progress converted to spline frame
        delta = projection.progress - spline_progress

        #projection.min_index gives
        spline: Spline = self.optimized_path[projection.min_index]

        # Find point along the spline for a given LOOKAHEAD
        point, _, _, x  = spline.along(self.LOOKAHEAD + delta, precision=20)

        curvature = frenet.get_curvature(spline,x) #overload?
        deriv = spline.first_der(x) # gradient at the point
        yaw = math.atan2(deriv,1) #calculate angle from the point 


        # publishing the data to controls
        msg = self.state_to_msg(point,curvature,yaw)
        self.carrot_publisher.publish(msg)
        self.get_logger().info('Publishing Raceline Path:')

    def state_to_msg(point, curvature, yaw):
        msg = CarROT()
        # geometry_msgs/Pose pose  # relative to car
        # float64 curvature
        #this is what controls is currently using from the carrot message
        msg.x = point[0]
        msg.y = point[1]
        #convert yaw to quarternion
        quaternion = axangle2quat([0,0,1],yaw)
        msg.pose.orientation.x = quaternion[0]
        msg.pose.orientation.y = quaternion[1]
        msg.pose.orientation.z = quaternion[2]
        msg.pose.orientation.w = quaternion[3]
        #msg.yaw = yaw
        msg.curvature = curvature
        return msg
    
    


def main(args=None):
    rclpy.init(args=args)

    lto = LTO()
    
    print("Starting Optimizer Node!")
    rclpy.spin(lto)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    lto.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
