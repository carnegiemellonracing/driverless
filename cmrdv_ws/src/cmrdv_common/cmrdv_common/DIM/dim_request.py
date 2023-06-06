import os
import rclpy
import asyncio

from std_msgs.msg import String
from rclpy.node import Node
from cmrdv_ws.src.cmrdv_common.cmrdv_common.heartbeat.heart import HeartbeatNode
from cmrdv_ws.src.cmrdv_common.cmrdv_common.CAN.can_types import * 

import cmrdv_ws.src.cmrdv_common.cmrdv_common.CAN.can_query as can_query

# TODO cannot find some canid definitions, confused.
VSM_HEARTBEAT = 'VSM_Heartbeat'
DIM_HEARTBEAT = 'DIM_Heartbeat'
DIM_REQUEST = 'DIM_Request'
CDC_MOTOR_DATA = 'CDC_MotorData'

class DIMStateNode(Node):
    """
    from carnegiemellonracing/stm32f413-drivers:
    /** @brief Standard CAN heartbeat. */
    typedef struct {
        uint8_t state;          /**< @brief Board state. */
        uint8_t error[2];       /**< @brief Error matrix. */
        uint8_t warning[2];     /**< @brief Warning matrix. */
    } cmr_canHeartbeat_t;

    Parameters
    -----------
    """

    def __init__(self):
        super().__init__('dim_request') 
        self.state_timer = self.create_timer(TX10HZ_PERIOD_S, self.send_dim_request)
        self.state_subscriber = self.create_subscription(
                String,
                "DIM_request",
                self.process_state_request,
                10
        )

        self.dim_state = {
            'vsm_req' : STATE.CMR_CAN_GLV_ON, 
            'gear' : GEAR.CMR_CAN_GEAR_SLOW,
            'gear_req' : GEAR.CMR_CAN_GEAR_SLOW
        }
        self.last_wake_time = self.get_clock().now()

    def state_get_vsm(self):
        """
        Returns the VSM State as read from CAN 

        Returns
        -------
        int
            Current VSM State as defined in `CAN/can_types.py`
        """
        vsm_heartbeat = asyncio.run(can_query.query_message(VSM_HEARTBEAT))
        return vsm_heartbeat['VSM_state']

    def vsm_req_valid(self, vsm, vsm_req):
        """
        Checks if the requested VSM state is allowed.
        Reference: https://github.com/carnegie-autonomous-racing/DIM/blob/main/state.c

        Parameters
        ----------
        vsm : int
            Current VSM state defined in `CAN/can_types.py`
        vsm_req : int
            VSM state being requested defined in `CAN/can_types.py`
        """
        if vsm == STATE.CMR_CAN_UNKNOWN:
            return (vsm_req == STATE.CMR_CAN_GLV_ON)
        elif vsm == STATE.CMR_CAN_GLV_ON:
            return (vsm_req == STATE.CMR_CAN_GLV_ON) or (vsm_req == STATE.CMR_CAN_HV_EN)
        elif vsm == STATE.CMR_CAN_HV_EN:
            return (vsm_req == STATE.CMR_CAN_GLV_ON) or (vsm_req == STATE.CMR_CAN_HV_EN) or (vsm_req == STATE.CMR_CAN_RTD)
        elif vsm == STATE.CMR_CAN_RTD:
            return (vsm_req == STATE.CMR_CAN_HV_EN) or (vsm_req == STATE.CMR_CAN_RTD)
        elif vsm == STATE.CMR_CAN_ERROR:
            return (vsm_req == STATE.CMR_CAN_GLV_ON)
        elif vsm == STATE.CMR_CAN_CLEAR_ERROR:
            return (vsm_req == STATE.CMR_CAN_GLV_ON)
        return False

    def state_vsm_up(self):
        """
        Handles VSM state up.
        Reference: https://github.com/carnegie-autonomous-racing/DIM/blob/main/state.c
        """
        vsm_state = self.state_get_vsm()
        # cancel state up 
        if self.dim_state['vsm_req'] < vsm_state:
            self.dim_state['vsm_req'] = vsm_state
            return
        
        vsm_req = STATE.CMR_CAN_GLV_ON if (vsm_state == STATE.CMR_CAN_UNKNOWN) or (vsm_state == STATE.CMR_CAN_ERROR) else vsm_state + 1

        # TODO: may need to lock for threading
        if self.vsm_req_valid(vsm_state, vsm_req):
            self.dim_state['vsm_req'] = vsm_req

    def state_vsm_down(self):
        """
        Handles VSM state down request.
        Reference: https://github.com/carnegie-autonomous-racing/DIM/blob/main/state.c
        """
        vsm_state = self.state_get_vsm()
        # cancel state down
        if self.dim_state['vsm_req'] > vsm_state:
            self.dim_state['vsm_req'] = vsm_state
            return

        motor_data = asyncio.run(can_query.query_message(CDC_MOTOR_DATA))
        speed_rpm = motor_data['CDC_motorSpeed']

        # only exit RTD if the car is near stopped
        if self.dim_state['vsm_req'] == STATE.CMR_CAN_RTD and speed_rpm > 5:
            return
        
        vsm_req = vsm_state - 1            # decrement state

        # TODO: may need to lock for threading
        if self.vsm_req_valid(vsm_state, vsm_req):
            self.dim_state['vsm_req'] = vsm_req

    def state_gear_switch(self):
        """
        Handles gear change button presses
        Reference: https://github.com/carnegie-autonomous-racing/DIM/blob/main/state.c
        """
        vsm_state = self.state_get_vsm()
        # Can only change gears in HV_EN and GLV_ON
        if (vsm_state != STATE.CMR_CAN_HV_EN):
            return
        
        if self.dim_state['gear'] != self.dim_state['gear_req']:
            return 

        gear_req = self.dim_state['gear'] + 1
        if gear_req < GEAR.CMR_CAN_GEAR_SLOW or gear_req >= GEAR.CMR_CAN_GEAR_LEN:
            gear_req = GEAR.CMR_CAN_GEAR_SLOW

        self.dim_state['gear_req'] = gear_req


    def update_req(self):
        """
        Updates state request to be consistent with VSM state.
        """
        self.dim_state['vsm_state'] = self.state_get_vsm()

    def update_gear(self):
        """
        Updates the gear to be the requested gear.
        """
        self.dim_state['gear'] = self.dim_state['gear_req']

        
    def send_dim_request(self):
        """
        Imitates the 10Hz CAN task for sending DIM request, nothing else. 
        TODO: for 10Hz tasks, need to send FSM info & etc.
        Reference: https://github.com/carnegie-autonomous-racing/DIM/blob/main/can.c
        """
        vsm_state = self.state_get_vsm()
        vsm_req = self.dim_state['vsm_req']
        gear = self.dim_state['gear']
        gear_req = self.dim_state['gear_req']

        self.get_logger().info(f"{vsm_state}, {self.dim_state}")
        
        if vsm_state == STATE.CMR_CAN_RTD and vsm_req == STATE.CMR_CAN_RTD:
            os.system('echo 0000 | sudo -S ip link set down can0')

        if (vsm_state != vsm_req) or (gear != gear_req):
            dim_request = {
                'DIM_requestState' : vsm_req,
                'DIM_requestGear' : gear_req
            }
            can_query.send_message(DIM_REQUEST, dim_request)
            # can_query.send_message("Auto_Heartbeat", {"Auto_state" : 4})

            self.update_gear()

    def process_state_request(self, msg):
        """
        Subsciber to update state request
        """
        request = msg.data
        if request == "up":
            self.state_vsm_up()
        if request == "down":
            self.state_vsm_down()
        if request == "gear":
            self.state_gear_switch()

        self.get_logger().info('Requested: "%s"' % msg.data)

    def check_timeout(self, last_received_ms, now_ms, threshold_ms=TIMEOUT_WARN_S):
        """
        Reference: cmr_canTimeout() 
        https://github.com/carnegiemellonracing/stm32f413-drivers/blob/6c63dee64cb3e9dc14c9a21a23728026d311a35e/CMR/can.c
        Checks if timeout has occurred 
        Parameters
        ----------
        last_received_ms : float
            Last receive timestamp, in milliseconds.
        now_ms : float
            Current timestamp, in milliseconds.
        threshold_ms : float
            Threshold period, in milliseconds.
        Returns
        -------
        A negative value (-1) if a timeout has occurred; otherwise 0.
        """
        # TODO: can we determine a default threshold; currently set to the timeout warning default
        release_ms = last_received_ms = threshold_ms

        # Current time overflowed; release did not. Timeout!
        if now_ms < last_received_ms and release_ms <= last_received_ms:
            return -1

        # Current time did not overflow; release time did. No timeout.
        if last_received_ms <= now_ms and release_ms < last_received_ms:
            return 0

        # Neither current nor release overflowed, or both have.
        # In either case, release less than current indicates timeout.
        if release_ms < now_ms: 
            return -1

        return 0


def main(args=None):
    rclpy.init(args=args)

    dim_state_node = DIMStateNode()

    rclpy.spin(dim_state_node)

    dim_state_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
