import rclpy
from rclpy.node import Node
# from interfaces.msg import ControlAction
from std_msgs.msg import Int32
import can
import time

BUSTYPE = 'pcan'
CHANNEL = 'PCAN_USBBUS1'
BITRATE = 500000
TIMER_HZ = 100
MAX_TORQUE = 2000 #this is completely made up
MAX_REQUEST = 200 #hypothetically real max is 255

class ActuatorNode(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.bus2 = can.interface.Bus(bustype=BUSTYPE, channel=CHANNEL, bitrate=BITRATE)
        # self.subscription = self.create_subscription(
        #     ControlAction,
        #     'control_action',  # Replace with the desired topic name
        #     self.callback,
        #     10  # Adjust the queue size as needed
        # )5
        self.int_subscription = self.create_subscription(
           Int32,
           '/swangle',
           self.swangle_callback,
           10
        )
        timer_period = 1/TIMER_HZ  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.even = True
        self.swangle = int(hex(2280)[2:], 16)
        self.accumulator = 0

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
        msg = can.Message(arbitration_id=0x134, data = [0x00ff & self.swangle, (0xff00 & self.swangle)>>8, 6,5,7,8,9,1], is_extended_id=False)
        self.bus2.send(msg)
        if (self.accumulator % 50 == 0):
            print(f"Message Sent on {self.bus2.channel_info}, data: {msg.data}")
        self.accumulator += 1
        # self.even = not self.even
    
    def callback(self,msg):
        (fl, fr, rl, rr) = (msg.torque_fl, msg.torque_fr, msg.torque_rl, msg.torque_rr)
        torque_avg = (fl+fr+rl+rr)/4
        torque_percent = min(1,max((torque_avg/MAX_TORQUE),-1)) #fucked clamping fix later

        #TODO: Read through CDC code and figure out what max regen request is to clamp in the negative direction

        torque_request = int(torque_percent*MAX_REQUEST)

        #where is this arbitration_id coming from? is this just our CAN ID?
        can_msg = can.Message(arbitration_id=0x134, data = [torque_request]*8, is_extended_id=False)
        self.bus2.send(can_msg)
    
    def swangle_callback(self,msg):
        self.swangle = int(hex(msg.data)[2:], 16)
    
def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = ActuatorNode()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()