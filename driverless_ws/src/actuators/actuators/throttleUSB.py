import rclpy
from rclpy.node import Node
from rclpy.time import Time
from interfaces.msg import ControlAction, ActuatorsInfo
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from std_msgs.msg import Int32
import numpy as np
import serial
import time
import math
import signal
import struct

BITRATE = 500000
TIMER_HZ = 100
MAX_TORQUE = 21 #this is completely made up
MAX_REQUEST = 255 #hypothetically real max is 255

ADC_BIAS = 2212
SLOPE = 34.5

CMDLINE_QOS_PROFILE = QoSProfile(
    depth=1,  # Set the queue depth
    reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,  # Set the reliability policy
    durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE  # Set the durability policy
)

class ActuatorNode(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        # self.ser = can.interface.Bus(bustype=BUSTYPE, channel=CHANNEL, bitrate=BITRATE,auto_reset=True)
        self.ser = serial.Serial("/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A10LIDBS-if00-port0", baudrate=9600, timeout=0.1)
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
        
        self.ser.reset_input_buffer()

        self.info_publisher = self.create_publisher(
            ActuatorsInfo,
            "/actuators_info",
            CMDLINE_QOS_PROFILE
        )

        # self.even = True
        self.swangle = int(hex(ADC_BIAS)[2:], 16)
        self.torque_request = 0
        msg = bytearray([0, 0,0,0])
        self.ser.write(msg) 
        self.orig_data_stamp = None
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
        # try:
            # throttle_msg = can.Message(arbitration_id=0x2D2, data = [self.torque_request]*8, is_extended_id=False)
            # self.ser.send(throttle_msg)
        
        #AIM recieves uint_8[8]
        data = (self.torque_request, self.swangle)
        print(data)
        x = bytearray()
        while(not x):
            x = self.ser.read(1)
            if x:
                print(x.hex())
            continue
        
        msg = struct.pack(">hh", data[0], data[1])
        # msg = bytearray([1,2,3,4])
        # msg = bytearray([7,7,7,7])
        
        self.ser.write(msg) 

        info = ActuatorsInfo()
        info.throttle_val = self.torque_request
        info.steering_adc = self.swangle
        info.latency_ms = -1
        info.header.stamp = self.get_clock().now().to_msg()

        print(f"Throttle Message Sent: {msg.hex()}, {msg}")
        if self.orig_data_stamp is not None:
            curr_time = self.get_clock().now()
            delta_nanos = curr_time.nanoseconds - self.orig_data_stamp.nanoseconds
            delta_ms = int(delta_nanos / 1e6)
            info.latency_ms = delta_ms
            print(f"Total Latency: {delta_ms} ms")

        self.info_publisher.publish(info)

            # steering_msg = can.Message(arbitration_id=0x134, data = [0x00ff & self.swangle, (0xff00 & self.swangle)>>8, 6,5,7,8,9,1], is_extended_id=False)
            # self.ser.send(steering_msg)
            # print(f"steering Message Sent: {steering_msg.data}")
            # print("----------------------")
        # except:
        #     print("tried to send")
        #     returnVal = self.ser.reset()
        #     if self.ser._is_shutdown:
        #         self.ser = can.interface.Bus(bustype=BUSTYPE, channel=CHANNEL, bitrate=BITRATE)
        #     else: self.ser.shutdown()
        #     print("Reset Return Value: ", returnVal)

    def callback(self,msg):
        (fl, fr, rl, rr) = (msg.torque_fl, msg.torque_fr, msg.torque_rl, msg.torque_rr)
        self.orig_data_stamp = Time.from_msg(msg.orig_data_stamp)
        
        print(f"fl: {fl} |fr: {fr} | rl: {rl} | rr: {rr}")
        #torque_avg = (fl+fr+rl+rr)/4
        torque_avg = (fl + fr)/2
        torque_percent = min(1,max((torque_avg/MAX_TORQUE),-1)) #fucked clamping fix later

        #TODO: Read through CDC code and figure out what max regen request is to clamp in the negative direction

        self.torque_request = int(torque_percent*127 + 128)
        print(self.torque_request) 

        desired_wheel_angle = np.degrees(msg.swangle) # convert from radians to degrees
        # desired_wheel_angle = msg.swangle
        clamped_wheel_angle = min(max(desired_wheel_angle, -19), 19)
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

    def sigint_handler(*args):
        print("ur mom closing down for business")
        # minimal_publisher.ser.close()
        # minimal_publisher.ser.shutdown()
        minimal_publisher.destroy_node()
        rclpy.shutdown()
    
    signal.signal(signal.SIGINT, sigint_handler)

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
