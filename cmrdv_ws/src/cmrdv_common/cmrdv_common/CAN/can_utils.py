import can
import signal
import asyncio
import typing

from can.interfaces.udp_multicast import UdpMulticastBus

def timeout_handler(signum, frame):
    raise TimeoutError("Timed out")

signal.signal(signal.SIGALRM, timeout_handler)

async def get_data(can_id, timeout: int):
    # print(can_id)
    # print(timeout)
    loop = asyncio.get_running_loop()
    with can.Bus(interface="socketcan", channel="can0") as bus:
        # bus = UdpMulticastBus(channel=UdpMulticastBus.DEFAULT_GROUP_IPv6) 
        reader = can.AsyncBufferedReader()
        logger = can.Logger("logfile.asc")
        notifier = can.Notifier(bus, [reader, logger], loop=loop)
        # print(can_id)
        # print(timeout)
        current_message: typing.Optional[can.Message] = None
        # print(current_message)
        signal.alarm(timeout)
        while current_message is None or current_message.arbitration_id != can_id:
            current_message = await reader.get_message()
        if current_message is None or current_message.arbitration_id != can_id:
            notifier.stop()
            raise Exception("Did not receive wanted message")
        # print(current_message)
        notifier.stop()
    return current_message

def send_data(can_id, data):
    message = can.Message(arbitration_id=can_id, data=data, check=True, is_extended_id=False)
    with can.Bus(interface="socketcan", channel="can0") as bus:
        bus.send(message)
    return
