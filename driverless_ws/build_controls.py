#!/usr/bin/python3

import argparse 
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build controls")
    parser.add_argument("-a", "--asserts", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-D", "--display", action="store_true")
    parser.add_argument("-c", "--data_collection", action="store_true")
    parser.add_argument("-e", "--export", action="store_true", help="export compile commands")
    parser.add_argument("-p", "--profile", action="store_true")

    args = parser.parse_args()
    command = "colcon build --packages-up-to controls --cmake-args"
    if args.asserts:
        command += " -DPARANOID=ON"
    if args.debug:
        command += " -DCMAKE_BUILD_TYPE=Debug"
    if args.display:
        command += " -DDISPLAY=ON"
    if args.export:
        command += " -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    if args.data_collection:
        command += " -DDATA=ON"
    if args.profile:
        command += "--profile"

    os.system(command)


