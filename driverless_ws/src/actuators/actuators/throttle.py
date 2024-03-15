import rclpy
from rclpy.node import Node
from interfaces.msg import ControlAction
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from interfaces.msg import ControlAction
from std_msgs.msg import Int32
import numpy as np
import can
import time
import math
import pdb

BUSTYPE = 'pcan'
CHANNEL = 'PCAN_USBBUS1'
BITRATE = 500000
TIMER_HZ = 100
MAX_TORQUE = 200 #this is completely made up
MAX_REQUEST = 255 #hypothetically real max is 255

ADC_BIAS = 2212
SLOPE = 34.5

CMDLINE_QOS_PROFILE = QoSProfile(
    depth=10,  # Set the queue depth
    reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,  # Set the reliability policy
    durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE  # Set the durability policy
)


class ActuatorNode(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.bus2 = can.interface.Bus(bustype=BUSTYPE, channel=CHANNEL, bitrate=BITRATE)
        

        self.subscription = self.create_subscription(
            ControlAction,
            '/control_action',  # Replace with the desired topic name
            self.callback,
            CMDLINE_QOS_PROFILE
        )
        # self.int_subscription = self.create_subscription(
        #    Int32,
        #    '/swangle',
        #    self.swangle_callback,
        #    10
        # )
        timer_period = 1/TIMER_HZ  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        # self.even = True
        self.swangle = int(hex(ADC_BIAS)[2:], 16)
        self.torque_request = 0
        # self.accumulator = 0

    def timer_callback(self):
        # to test regen: send full throttle, then full negative for regen
        # then send full throttle, and like, half regen --> should see slightly less decceleration
        # should reconfigure this whole thing to pipe floats to AIM over CAN, so we dont have to convert
        # from float -> uint8 -> float
        # num = 0 if self.even else 255
        # hypothetically should not need to convert to bytearray,but just incase, that is a thing we *can* do
        # swangle = -1
        # steering_bytearr = swangle.to_bytes()

        # first element in data array is throttle(uint8), next 2 is steering (int16)
        #where is this arbitration_id coming from? is this just our CAN ID?
        throttle_msg = can.Message(arbitration_id=0x2D2, data = [self.torque_request]*8, is_extended_id=False)
        self.bus2.send(throttle_msg)
        print(f"Throttle Message Sent: {throttle_msg.data}")

        steering_msg = can.Message(arbitration_id=0x134, data = [0x00ff & self.swangle, (0xff00 & self.swangle)>>8, 6,5,7,8,9,1], is_extended_id=False)
        self.bus2.send(steering_msg)
        print(f"steering Message Sent: {steering_msg.data}")
        # print(self.swangle)
        print("----------------------")
    
    def callback(self,msg):
        #TODO: SHIFT TORQUE VALUE TO BE CENTERED AROUND 128
        (fl, fr, rl, rr) = (msg.torque_fl, msg.torque_fr, msg.torque_rl, msg.torque_rr)
        print(f"fl: {fl} |fr: {fr} | rl: {rl} | rr: {rr}")
        torque_avg = (fl+fr+rl+rr)/4
        torque_percent = min(1,max((torque_avg/MAX_TORQUE),-1)) #fucked clamping fix later

        #TODO: Read through CDC code and figure out what max regen request is to clamp in the negative direction

        self.torque_request = int(torque_percent*127 + 128)
        print(self.torque_request) 

        desired_wheel_angle = np.degrees(msg.swangle) # convert from radians to degrees
        clamped_wheel_angle = min(max(desired_wheel_angle, -20), 20)
        adc_val = int(SLOPE * clamped_wheel_angle + ADC_BIAS)
        self.swangle = int(hex(adc_val)[2:], 16)
        print(f"swangle: {self.swangle}")

        
    
    def swangle_callback(self,msg):
        desired_wheel_angle = msg.data
        adc_val = int(34.5 * desired_wheel_angle + 2212)
        self.swangle = int(hex(adc_val)[2:], 16)
    
def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = ActuatorNode()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    # minimal_publisher.bus2.shutdown()
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()