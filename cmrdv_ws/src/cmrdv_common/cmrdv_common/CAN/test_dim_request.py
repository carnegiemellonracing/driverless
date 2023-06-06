import can_query
from can_types import * 
import time
import numpy as np
import cv2
import os
DIM_REQUEST = 'DIM_Request'
vsm_req = STATE.CMR_CAN_GLV_ON
gear_req = GEAR.CMR_CAN_GEAR_FAST
dim_state = {
            'vsm_req' : vsm_req, 
            'gear' : GEAR.CMR_CAN_GEAR_FAST,
            'gear_req' : gear_req
        }
dim_request = {
                'DIM_requestState' : vsm_req,
                'DIM_requestGear' : gear_req
            }
while(True):
    cv2.imshow('urmom', np.zeros((256, 256), dtype=np.uint8))
    letter = cv2.waitKey(100)
    if letter == ord('u'):
        vsm_req += 1
        os.environ['DIM_STATE'] = str(vsm_req)
    if letter == ord('d'):
        vsm_req -= 1
        os.environ['DIM_STATE'] = str(vsm_req)
    
    dim_state = {
            'vsm_req' : vsm_req, 
            'gear' : GEAR.CMR_CAN_GEAR_FAST,
            'gear_req' : gear_req
        }
    dim_request = {
                'DIM_requestState' : vsm_req,
                'DIM_requestGear' : gear_req
            }
    
    can_query.send_message(DIM_REQUEST, dim_request)
    print(f'time: {time.time()}    | vsm_req: {vsm_req}')
    time.sleep(0.1)
