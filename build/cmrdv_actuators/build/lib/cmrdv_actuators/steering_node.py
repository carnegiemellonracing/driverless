from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.collection_config import BEST_EFFORT_QOS_PROFILE
import rclpy
from rclpy.node import Node
from rclpy.subscription import Subscription

from cmrdv_ws.src.cmrdv_actuators.cmrdv_actuators.driver_board import DriverBoard
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.controls_config import CONTROL_ACTION_TOPIC
from cmrdv_interfaces.msg import DataFrame, ControlAction

from std_msgs.msg import Int64

from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.actuators_config import (STEER_CHANNEL,
                                                                            STEER_TOPIC,
                                                                            STEER_MIN_ANGLE, 
                                                                            STEER_MAX_ANGLE,
                                                                            STEER_MIN_VAL,
                                                                            STEER_MID_VAL,
                                                                            STEER_MAX_VAL,
                                                                            angle_to_PWM)
import math
import numpy as np

class SteeringNode(Node):
    def __init__(self):
        super().__init__('steering_node')

        
        self.driver_board = DriverBoard.get_instance()
        
        # Subscribe to the PWM signal topic
        #self.sub = self.create_subscription(Int16, 'pwm_signal', self.pwm_callback, 10)
        
        # subscribes to controls topic
        # calls controls_callback function upon receiving a new published value
        self.brakes_subscriber = self.create_subscription(
            ControlAction,
            CONTROL_ACTION_TOPIC,
            self.steering_controls_callback,
            qos_profile=BEST_EFFORT_QOS_PROFILE,
        )
        
        self.driver_board.steering_set_pwm_channel(STEER_CHANNEL, STEER_MID_VAL)
        
        #TEST TOPIC CHANGE LATER
        self.steering_subscriber = self.create_subscription(
            Int64,
            STEER_TOPIC,
            self.steering_callback,
            qos_profile=BEST_EFFORT_QOS_PROFILE,
        )

    def get_steering_pwm(self, angle):
        """
        Calculates the pwm signal based off of a linear angle -> pwm signal model
        """
        pwm = angle_to_PWM(angle)
        return np.clip(int(pwm), STEER_MIN_VAL, STEER_MAX_VAL, dtype=np.int16)

    def steering_controls_callback(self, msg):
        """
        Callback function for controls action topic, using swangle.
        """
        swangle = msg.swangle
        pwm_val = self.get_steering_pwm(swangle)

        print(f"swangle: {swangle:.3f} val: {pwm_val:.3f}")
        self.get_logger().info(f"swangle: {swangle:.3f} val: {pwm_val:.3f}")
        # self.get_logger().info(f"PWMVAL: {pwm_val}")
        self.driver_board.steering_set_pwm_channel(STEER_CHANNEL, pwm_val)

    def steering_callback(self, msg):
        """
        Callback function for testing only steering PWM signals. Topic publishes PWM signals.
        """
        pwm_val = msg.data
        if (pwm_val < STEER_MIN_VAL):
            pwm_val = STEER_MIN_VAL
        
        elif (pwm_val > STEER_MAX_VAL):
            pwm_val = STEER_MAX_VAL

        print("val: ", pwm_val)
        self.driver_board.steering_set_pwm_channel(STEER_CHANNEL, pwm_val)


def main(args=None):
    rclpy.init(args=args)

    steering_node = SteeringNode()

    rclpy.spin(steering_node)
    steering_node.destroy_node()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()
