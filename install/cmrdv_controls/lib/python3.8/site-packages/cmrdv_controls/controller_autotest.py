import numpy as np
import rclpy

from cmrdv_interfaces.msg import ControlAction
from cmrdv_ws.src.cmrdv_common.cmrdv_common.DIM.dim_heartbeat import HeartbeatNode
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.controls_config import CONTROL_ACTION_TOPIC
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.collection_config import BEST_EFFORT_QOS_PROFILE

# from cmrdv_common.heartbeat.heart import HeartbeatNode
# from cmrdv_common.config.controls_config import CONTROL_ACTION_TOPIC
# from cmrdv_common.config.collection_config import BEST_EFFORT_QOS_PROFILE

# TODO: add autotest controller to overall necessary nodes for heartbeat
class AutotestController(HeartbeatNode):
    """
    Controller node that performs autonomous test: 
        - drive straight
        - turn right
        - turn left
        - brake
    Publishes these control actions

    Parameters
    -----------
    delay : float
        Number of seconds spent on each action
    """
    def __init__(self, delay=10):
        super().__init__('autotest_controller')
        
        # pre defined actions to go straight, turn left, turn right, brake
        # TODO: confirm that 0.0 is the straight position, should be documented
        self.actions = [
            [[2.5], [0.0]],                 # go straight, wheel speed: 2.5 rad/s swangle: 0.0
            [[2.5], [-25.0]],               # turn left
            [[2.5], [25.0]],                # turn right
            [[2.5], [0.0]],                 # return to straight
            [[0.0], [0.0]]                  # brakes
        ]

        self.control_action_publisher = self.create_publisher(
            ControlAction,
            CONTROL_ACTION_TOPIC,
            BEST_EFFORT_QOS_PROFILE
        )

        self.send_control_action_timer = self.create_timer(
            delay,                          # send control action every `delay` seconds
            self.send_control_action        
        )
        self.action_number = 0

        # tuple containing desired point, yaw, and curvature
        # if None, we are waiting for first carrot and should not take any
        # control action
        self.setpoint = None
        self.velocity = None
        self.yaw_dot = None

    def send_control_action(self):
        # TODO: add a check for the parameter that shows it is active node
        if not self.alive(): 
            return 
        if self.action_number >= len(self.actions):
            return

        control_action = self.actions[self.action_number]
        self.control_action_publisher.publish(
            self.control_action_array_to_msg(control_action)
        )

        self.action_number += 1

        print(f'CONTROL ACTION: #{self.action_number} {control_action}')
        print('----------')


    def vel_subscription_callback(self, msg):
        """Subscriber to car position topic. Creates new setpoint based on path
        planning data and uses feedfoward and lqr to generate control action.
        Runs as fast as new position data is generated.
        """

        if self.setpoint is None:
            return

        self.velocity = np.array([msg.twist.linear.x, msg.twist.linear.y])
        self.yaw_dot = msg.twist.angular.z

        print('---------')
        print(f'VELOCITY: {self.velocity}')
        print(f'YAW_DOT: {self.yaw_dot}')

        inertial_state = self.get_state()

        self.lqr.current_state = inertial_state
        print(f'INERTIAL STATE: {inertial_state}')

        self.lqr.desired_state = self.setpoint
        self.feedforward.desired_state = self.setpoint

        feedforward_ctrl_action = self.feedforward.estimate_control_action()
        feedback_ctrl_action = self.lqr.get_control_action()

        print(f'FEEDFRWD ACTION: {feedforward_ctrl_action}')
        print(f'FEEDBACK ACTION: {feedback_ctrl_action}')
        
        control_action = np.add(
            feedforward_ctrl_action, 
            feedback_ctrl_action
        )

        self.control_action_publisher.publish(
            self.control_action_array_to_msg(control_action)
        )

        print(f'TOTAL CONTROL ACTION: {control_action}')
        print('----------')


    def get_state(self):
        """Transforms state message to 6x1 ndarray containing
        [[x],[y],[yaw],[xdot],[ydot],[yawdot]]
        """
        return np.array([
            [0],
            [0],
            [0],
            [self.velocity[0]],
            [self.velocity[1]],
            [self.yaw_dot]
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

    controller = AutotestController()
    rclpy.spin(controller)

    controller.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()