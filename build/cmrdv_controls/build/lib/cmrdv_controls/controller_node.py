import numpy as np
import rclpy
import math

from rclpy.node import Node
from rclpy.subscription import Subscription
from cmrdv_ws.src.cmrdv_controls.cmrdv_controls.controller_tasks import LQR, Feedforward, SetpointGenerator
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.controls_config import CONTROL_ACTION_TOPIC, GAIN_MATRIX, GOAL_SPEED, CARROT_AVG_HISTORY, CONTROLLER_DEADZONE
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.planning_config import PAIRROT_TOPIC
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.collection_config import BEST_EFFORT_QOS_PROFILE
from cmrdv_ws.src.cmrdv_common.cmrdv_common.DIM.dim_heartbeat import HeartbeatNode
from cmrdv_interfaces.msg import PairROT, ControlAction


class MovingAverage(object):
    def __init__(self, n, init):
        self.history = np.tile(init, n)

    def log_sample(self, sample):
        self.history = np.concatenate((self.history[:, 1:], sample), axis=1)

    def get_avg(self):
        return np.mean(self.history, axis=1, keepdims=True)


class Controller(HeartbeatNode):
    """Controller node. Subscribes to updates in car state, and publishes
    control actions.
    """

    def __init__(self):
        super().__init__('controller')

        self.control_action_publisher = self.create_publisher(
            ControlAction,
            CONTROL_ACTION_TOPIC,
            BEST_EFFORT_QOS_PROFILE
        )

        self.parrot_subscriber: Subscription = self.create_subscription(
            PairROT,
            PAIRROT_TOPIC,
            self.parrot_subscription_callback,
            BEST_EFFORT_QOS_PROFILE
        )

        self.lqr = LQR()
        self.lqr.gain_matrix = GAIN_MATRIX

        self.feedforward = Feedforward()
        self.setpoint_generator = SetpointGenerator()

        self.near_curvatire_moving_avg = MovingAverage(CARROT_AVG_HISTORY, [[0]])
        self.far_setpoint_moving_avg = MovingAverage(CARROT_AVG_HISTORY, np.zeros(shape=(6,1)))

    def parrot_subscription_callback(self, msg):
        """BOB controller 

        bang-off-bang
        """

        carrot = msg.near
        carrot_angle = np.rad2deg(math.atan2(-carrot.x, carrot.y))
        swangle = 0.
        if carrot_angle > CONTROLLER_DEADZONE:
            swangle = 10.
        if carrot_angle < -CONTROLLER_DEADZONE:
            swangle = -10.

        res = Controller.control_action_array_to_msg(np.array([[0],[swangle]]))
        self.control_action_publisher.publish(res)

    #     near_carrot = msg.near
    #     far_carrot = msg.far

    #     print('---------------')
    #     print(f'NEAR CARROT:')
    #     print(f'X: {near_carrot.x} Y: {near_carrot.y} YAW: {near_carrot.yaw} CURVATURE: {near_carrot.curvature}')
        
    #     self.near_curvatire_moving_avg.log_sample([[near_carrot.curvature]])
    #     near_curvature_avg =self.near_curvatire_moving_avg.get_avg()[0, 0]
        
    #     print(f'MOVING AVG CURVATURE: {near_curvature_avg}')

    #     print('---------------')
    #     print(f'FAR CARROT:')
    #     print(f'X: {far_carrot.x} Y: {far_carrot.y} YAW: {far_carrot.yaw} CURVATURE: {far_carrot.curvature}')

    #     far_setpoint = self.setpoint_generator.get_setpoint(
    #         GOAL_SPEED, np.array([0, 0]), np.array([far_carrot.x, far_carrot.y]), 
    #         0, far_carrot.yaw, far_carrot.curvature
    #     )
    #     print(f'FAR SETPOINT: {far_setpoint}')

    #     self.far_setpoint_moving_avg.log_sample(far_setpoint)
    #     far_setpoint_avg = self.far_setpoint_moving_avg.get_avg()

    #     print(f'MOVING AVG FAR SETPOINT: {far_setpoint_avg}')

    #     print('---------')


    #     inertial_state = self.get_state()

    #     self.lqr.current_state = inertial_state
    #     print(f'INERTIAL STATE: {inertial_state}')

    #     self.lqr.desired_state = far_setpoint_avg
    #     self.feedforward.curvature = near_curvature_avg

    #     feedforward_ctrl_action = self.feedforward.estimate_control_action()
    #     feedback_ctrl_action = self.lqr.get_control_action()

    #     self.get_logger().info(f'FEEDFRWD ACTION: {feedforward_ctrl_action}')
    #     self.get_logger().info(f'FEEDBACK ACTION: {feedback_ctrl_action}')
        
    #     control_action = np.add(
    #         0*feedforward_ctrl_action, 
    #         feedback_ctrl_action
    #     )

    #     self.control_action_publisher.publish(
    #         Controller.control_action_array_to_msg(control_action)
    #     )

    #     self.get_logger().info(f'TOTAL CONTROL ACTION: {control_action}')

    def get_state(self):

        """Transforms state message to 6x1 ndarray containing
        [[x],[y],[yaw],[xdot],[ydot],[yawdot]]
        """

        # degrees

        return np.array([
            [0],
            [0],
            [0],
            [0],
            [GOAL_SPEED],
            [0]
        ])

    @staticmethod
    def control_action_array_to_msg(control_action):
        """Transforms 2x1 control action array ([[wheel speed], [swangle]]) to
        ros message.
        """

        msg = ControlAction()
        msg.wheel_speed = control_action[0][0]
        msg.swangle = control_action[1][0]
        
        return msg
    
def main():
    rclpy.init()

    controller = Controller()
    rclpy.spin(controller)

    controller.destroy_node()
    rclpy.shutdown()
