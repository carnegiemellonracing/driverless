import os 
import time
import struct
import asyncio
import subprocess

import rclpy
from rclpy.node import Node

from cmrdv_ws.src.cmrdv_common.cmrdv_common.heartbeat.heart import HeartbeatNode
import cmrdv_ws.src.cmrdv_common.cmrdv_common.CAN.can_query as can_query
from cmrdv_ws.src.cmrdv_common.cmrdv_common.CAN.can_types import * 
from cmrdv_ws.src.cmrdv_common.cmrdv_common.CAN.can_types import ERROR_STATE

VSM_HEARTBEAT = 'VSM_Heartbeat'
DIM_HEARTBEAT = 'DIM_Heartbeat'

class DIMHeartbeatNode(Node):
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
        super().__init__('dim_heartbeat') 

        self.send_timer = self.create_timer(TX10HZ_PERIOD_S, self.send_heartbeat)
        self.read_timer = self.create_timer(TX10HZ_PERIOD_S, self.process_heartbeat)

        self.dim_heartbeat = {
            'DIM_state' : STATE.CMR_CAN_UNKNOWN, 
            'DIM_ERR_VSMTimeout' : ERROR.CMR_CAN_ERROR_NONE
        }
        self.last_wake_time = self.get_clock().now()

    def send_heartbeat(self):
        can_query.send_message(DIM_HEARTBEAT, self.dim_heartbeat)

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
        release_ms = last_received_ms + rclpy.time.Duration(seconds=threshold_ms)

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

    def process_heartbeat(self):
        """
        Reference: DIM sendHeartbeat() 
        https://github.com/carnegie-autonomous-racing/DIM/blob/main/can.c 
        If the can reception has timed out, will process the previous heartbeat, but change state to ERROR. 
        In theory, should panic system when this happens. 

        """
        
        vsm_heartbeat = asyncio.run(can_query.query_message(VSM_HEARTBEAT))

        # cmr_canWarn_t warning = CMR_CAN_WARN_NONE;
        # cmr_canError_t error = CMR_CAN_ERROR_NONE;
        state = vsm_heartbeat['VSM_state']

        # if state == STATE.CMR_CAN_RTD:
        #     os.system('echo 0000 | sudo -S ip link set down can1')
        #     self.destroy_node()

        warning = WARN.CMR_CAN_WARN_NONE
        error = ERROR.CMR_CAN_ERROR_NONE

        current_wake_time = self.get_clock().now()

        # if (cmr_canRXMetaTimeoutError(heartbeatVSMMeta, lastWakeTime) < 0) {
        #     error |= CMR_CAN_ERROR_VSM_TIMEOUT;
        # }
        if self.check_timeout(self.last_wake_time, current_wake_time, TIMEOUT_ERROR_S) < 0:
            error |= ERROR.CMR_CAN_ERROR_VSM_TIMEOUT

        # if (cmr_canRXMetaTimeoutWarn(heartbeatVSMMeta, lastWakeTime) < 0) {
        #     warning |= CMR_CAN_WARN_VSM_TIMEOUT;
        # }
        if self.check_timeout(self.last_wake_time, current_wake_time, TIMEOUT_WARN_S) < 0:
            warning |= WARN.CMR_CAN_WARN_VSM_TIMEOUT

        # reset the wake times
        self.last_wake_time = current_wake_time

        # create DIM Heartbeat
        dim_heartbeat = {
            'DIM_state' : state, 
            'DIM_ERR_VSMTimeout' : error 
        }

        self.dim_heartbeat = dim_heartbeat

def main(args=None):
    rclpy.init(args=args)

    dim_heartbeat_node = DIMHeartbeatNode()

    rclpy.spin(dim_heartbeat_node)

    dim_heartbeat_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
