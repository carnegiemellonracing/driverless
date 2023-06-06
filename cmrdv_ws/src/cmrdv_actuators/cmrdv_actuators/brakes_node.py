
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.collection_config import BEST_EFFORT_QOS_PROFILE
import rclpy
from rclpy.node import Node
from rclpy.subscription import Subscription

from cmrdv_ws.src.cmrdv_actuators.cmrdv_actuators.driver_board import DriverBoard
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.controls_config import CONTROL_ACTION_TOPIC
from cmrdv_interfaces.msg import DataFrame, ControlAction
from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.collection_config import RELIABLE_QOS_PROFILE

from cmrdv_interfaces.msg import Brakes

from std_msgs.msg import Int64

from cmrdv_ws.src.cmrdv_common.cmrdv_common.config.actuators_config import (BRAKES_CHANNEL_1, 
                                                                  BRAKES_CHANNEL_2, 
                                                                  BRAKES_STOP_VAL, 
                                                                  MAX_BRAKES, 
                                                                  EN_PIN, 
                                                                  ENB_PIN,
                                                                  DELTA_THRESH,
                                                                  MAX_EXTEND,
                                                                  BRAKES_STATUS_TOPIC,
                                                                  BRAKES_TOPIC,
                                                                  ADC_BUFFER,
                                                                  MAX_ADC_VAL,
                                                                  MIN_ADC_VAL)

class BrakesNode(Node):
    """
    Attributes
    ----------
    actuating : bool
        Whether or not the brakes are moving
    
    """
    def __init__(self):
        super().__init__('brakes_node')
        self.driver_board = DriverBoard.get_instance()

        #Brakes driver carrier: set en and enb to to high and low respectively
        self.driver_board.jetson_setup_pin(EN_PIN, True)
        self.driver_board.jetson_setup_pin(ENB_PIN, False)

        self.prev_wheel_speed = 0
        self.potentiometer_pos = 0

        self.actuating = False
        self.last_actuated = self.get_clock().now()
        
        # subscribes to controls topic
        # calls controls_callback function upon receiving a new published value
   
        # self.brakes_subscriber = self.create_subscription(
        #     ControlAction,
        #     CONTROL_ACTION_TOPIC,
        #     self.controls_callback,
        #     10,
        # )
        
        #TEST TOPIC CHANGE LATER
        self.brakes_subscriber = self.create_subscription(
            Int64,
            BRAKES_TOPIC,
            self.brakes_callback,
            qos_profile=BEST_EFFORT_QOS_PROFILE,
        )

        self.status_publisher = self.create_publisher(
            Brakes,
            BRAKES_STATUS_TOPIC,
            RELIABLE_QOS_PROFILE
        )

    def retract(self):
        print("Retracting...")
        self.driver_board.set_pwm_channel(BRAKES_CHANNEL_1, BRAKES_STOP_VAL)
        self.driver_board.set_pwm_channel(BRAKES_CHANNEL_2, MAX_BRAKES)

        print("Retracted!")

    def extend(self):
        print("Extending...")
        self.driver_board.set_pwm_channel(BRAKES_CHANNEL_1, MAX_BRAKES)
        self.driver_board.set_pwm_channel(BRAKES_CHANNEL_2, BRAKES_STOP_VAL)

        print("Extended!")
        
    def stopMotor(self):
        print("Stopping Motor")
        self.driver_board.set_pwm_channel(BRAKES_CHANNEL_1, BRAKES_STOP_VAL)
        self.driver_board.set_pwm_channel(BRAKES_CHANNEL_2, BRAKES_STOP_VAL)
        print("Motor stopped")

    def brakes_callback(self, msg):
        """
        If it receives a controls msg
        """
        wheel_speed = msg.data
        # wheel_speed = msg.wheel_speed

        adc_val = self.driver_board.get_adc_val()
        print("adc val: ", adc_val)
        # toggling self.actuation for throttle 
        if adc_val <= MIN_ADC_VAL or adc_val >= MAX_ADC_VAL:
            print("at max/min")
            self.actuating = False
        # determining whether or not to actuate 
        if MIN_ADC_VAL - ADC_BUFFER < adc_val < MAX_ADC_VAL:
            if wheel_speed - self.prev_wheel_speed > DELTA_THRESH:
                if self.potentiometer_pos < MAX_EXTEND: #max_extend is config value
                    # do logic to extend the linear actuator
                    self.extend()
                    self.actuating = True

            elif wheel_speed - self.prev_wheel_speed < DELTA_THRESH:
                # do logic to retract linear actutator
                self.retract()
                self.actuating = True

        self.prev_wheel_speed = wheel_speed
        self.brakes_status_pub()

    def brakes_status_pub(self):
        brake = Brakes()
        brake.braking = self.actuating

        if (self.actuating):
            self.last_actuated = self.get_clock().now()
            brake.last_fired = self.last_actuated.to_msg()
        else:
            brake.last_fired = self.last_actuated.to_msg()

        self.status_publisher.publish(brake)


def main(args=None):
    rclpy.init(args=args)

    brakes_node = BrakesNode()

    rclpy.spin(brakes_node)
    brakes_node.destroy_node()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()
