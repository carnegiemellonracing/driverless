#!/usr/bin/python3

import argparse 
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build point to pixel")

    args = parser.parse_args()
    command = "colcon build --cmake-clean-cache --packages-up-to point_to_pixel --cmake-args"

    os.system(command)


