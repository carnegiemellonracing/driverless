import can_query as can_query
from can_types import * 
import time
import os

DIM_HEARTBEAT = 'DIM_Heartbeat'
while(True):
    dim_heartbeat = {
                 'DIM_state' : int(os.getenv("DIM_STATE")),
                 'DIM_ERR_VSMTimeout' : ERROR.CMR_CAN_ERROR_NONE
    }
    print(f'time: {time.time()}    | vsm_req: {int(os.getenv("DIM_STATE"))}')
    can_query.send_message(DIM_HEARTBEAT, dim_heartbeat)
    time.sleep(0.01)
