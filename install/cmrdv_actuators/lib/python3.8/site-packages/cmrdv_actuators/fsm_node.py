import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.subscription import Subscription

from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.collection_config import BEST_EFFORT_QOS_PROFILE
from cmrdv_ws.src.cmrdv_actuators.cmrdv_actuators.driver_board import DriverBoard
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.controls_config import CONTROL_ACTION_TOPIC
from cmrdv_interfaces.msg import DataFrame, ControlAction, Brakes

from std_msgs.msg import Int64
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.actuators_config import (BRAKES_STATUS_TOPIC, 
                                                                            FSM_TOPIC, 
                                                                            FSM_SENSOR1_CHANNEL,
                                                                            FSM_SENSOR2_CHANNEL,
                                                                            FSM_START_VAL,
                                                                            FSM_STOP_VAL, 
                                                                            FSM_MAX_VAL,
                                                                            FSM_MIN_VAL)

class FSMSensorController(Node):
    """
    Controls two FSM sensors.
    """

    def __init__(self):
        super().__init__('fsm_node')
        self.driver_board = DriverBoard.get_instance()
        self.fsm_sensor1 = self.driver_board.get_pwm_channel(FSM_SENSOR1_CHANNEL)
        self.fsm_sensor2 = self.driver_board.get_pwm_channel(FSM_SENSOR2_CHANNEL)

        self.brake_actuating = False
        self.brake_last_actuated = self.get_clock().now()

        #start pwm cycle
        print("starting FSM")
        self.update_duty_cycle(FSM_START_VAL)

        #TODO: find correct topic to recieve pwm signals
#        self.control_action_subscription = self.create_subscription(
#                ControlAction,
#                CONTROL_ACTION_TOPIC,
#                self.controls_fsm_callback,
#                qos_profile=BEST_EFFORT_QOS_PROFILE)

        #TODO: find correct topic to recieve pwm signals
        self.control_action_subscription = self.create_subscription(
                Int64,
                FSM_TOPIC,
                self.fsm_callback,
                qos_profile=BEST_EFFORT_QOS_PROFILE)
        

        
        self.brake_status_subscription = self.create_subscription(
                Brakes,
                BRAKES_STATUS_TOPIC,
                self.brake_status_callback,
                qos_profile=BEST_EFFORT_QOS_PROFILE)
    
  
    def update_duty_cycle(self, val):
        
        if val > FSM_MAX_VAL:
            print("Value was too high. Setting it to Max value")
            val = FSM_MAX_VAL

        if val < FSM_MIN_VAL:
            print("value was too low. Setting it to Min value")
            val = FSM_MIN_VAL
        
        try:
            print("setting sensor1 to: ", val)
            self.driver_board.set_pwm_channel(FSM_SENSOR1_CHANNEL, val)
            print("setting sensor2 to: ", val)
            self.driver_board.set_pwm_channel(FSM_SENSOR2_CHANNEL, val)

        except Exception as e:
            print(e)
            print('Exception caught')
            self.terminateFSM()

    def controls_fsm_callback(self, msg):
        wheel_speed = msg.wheel_speed
        if not self.brake_actuating:
            #wait half a second
            if self.get_clock().now() - self.brake_last_actuated > rclpy.time.Duration(seconds=0.5):
                self.update_duty_cycle(wheel_speed)
        else: 
            self.update_duty_cycle(0)

    def controls_fsm_callback(self, msg):
        print(msg.wheel_speed)
        if not self.brake_actuating:
            #wait half a second
            if self.get_clock().now() - self.brake_last_actuated > rclpy.time.Duration(seconds=0.5):
                self.update_duty_cycle(9000)
        else: 
            self.update_duty_cycle(0)



    def get_wheel_speed_pwm(self, wheel_speed):
        pass

    def fsm_callback(self, msg):
        #TODO: find correct conversion between wheel speed and duty cycle
        print(self.brake_actuating)
        if not self.brake_actuating:
            #wait half a second
            if self.get_clock().now() - self.brake_last_actuated > rclpy.time.Duration(seconds=0.5):
                self.update_duty_cycle(msg.data)
        else: 
            self.update_duty_cycle(0)

    def brake_status_callback(self, msg):
        self.brake_actuating = msg.braking
        self.brake_last_actuated = Time.from_msg(msg.last_fired)


    def terminateFSM(self):
        print("Terminating FSM...")

        self.driver_board.set_pwm_channel(FSM_SENSOR1_CHANNEL, FSM_STOP_VAL)
        self.driver_board.set_pwm_channel(FSM_SENSOR2_CHANNEL, FSM_STOP_VAL)
        
        self.driver_board.kill()
        print("FSM Terminated")

def main(args=None):
    rclpy.init(args=args)

    fsm_controller = FSMSensorController()

    rclpy.spin(fsm_controller)

    # Destroy the node explicitly
    fsm_controller.destroy_node()
    rclpy.shutdown()
