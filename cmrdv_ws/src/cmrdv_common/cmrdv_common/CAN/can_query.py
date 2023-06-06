import sys
import asyncio
import time
import os
import cantools
import cmrdv_ws.src.cmrdv_common.cmrdv_common.CAN.can_utils as can_utils
from ament_index_python.packages import get_package_share_directory

db = cantools.database.load_file(os.path.join(os.path.dirname(__file__), "CMR_19e.dbc"))

lookup_dict = {}

MAX_TIMEOUT : int = 20000

for message in db.messages:
    lookup_dict[message.name] = (message.frame_id, message.cycle_time if message.cycle_time else MAX_TIMEOUT // 10)
    for signal in message.signals:
        lookup_dict[signal.name] = (message.frame_id, message.cycle_time)


async def query_message(message: str):
    """
    Parameters
    ----------
    message : str
        String corresponding to CAN ID. In the dbc file, it will look like 
        BO (some number) (message) (can id) (misc)
    """
    if message not in lookup_dict:
        raise Exception(f"No {message} found in {lookup_dict.keys()}")
    can_id, cycle_time = lookup_dict[message]
    timeout: int = min(cycle_time * 10, 20000)
    can_data = await can_utils.get_data(can_id=can_id, timeout=timeout)
    return db.decode_message(can_id, can_data.data, decode_containers=True)

def send_message(message : str, data : dict):
    """
    Send messages to CAN in one specific CAN ID

    Parameters
    ----------
    message : str
        String corresponding to CAN ID 
        BO (some number) (message) (can id) (misc)
    data : dict 
        Dictionary with keys corresponding to the second arguments of lines in DBC
        SG_ (key of arg) (misc)
    """
    if message not in lookup_dict:
        raise Exception(f"No {message} found in {lookup_dict.keys()}")
    can_id, _ = lookup_dict[message]
    can_data = db.encode_message(can_id, data, strict=True)
    can_utils.send_data(can_id=can_id, data=can_data)

async def test_query():
    while True:
        for data in asyncio.as_completed([query_message(message.name) for message in db.messages]):
            data = await data
            print(data)

if __name__ == '__main__':
    message = sys.argv[1]
    time_fired = time.time()
    while True:
        data = asyncio.run(query_message(message))
        if (True):
            print(data)
            time_fired = time.time()
