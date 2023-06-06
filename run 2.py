gitimport sys
import traceback
from argparse import ArgumentParser

from fsdv.mcl import MainControlLoop

def parse_opt():
    parser = ArgumentParser()
    parser.add_argument('--sim', action='store_true')
    opt = parser.parse_args()
    return opt

def handle_shutdown(mcl):
    while True: 
        user_input = input('Next move? ')
        if user_input == 'terminate': 
            mcl.terminate()
        if user_input == 'initialize': 
            mcl.initialize()
        if user_input == 'execute': 
            break 
        if user_input == 'exit':
            sys.exit(0)

if __name__ == "__main__":
    arg = parse_opt()
    MCL = MainControlLoop(arg.sim)
    while True: 
        try:
            MCL.execute()
        except KeyboardInterrupt:
            print("\nShutdown requested...exiting")
            handle_shutdown(MCL)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            sys.exit(0)
        finally: 
            pass
