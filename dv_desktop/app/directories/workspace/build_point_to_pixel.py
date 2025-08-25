from utils.types import Module
import subprocess

def _build_point_to_pixel_command():
    subprocess.run(["python3","build_point_to_pixel.py"], cwd = "../driverless_ws")


BuildP2PModule = Module(
    title="Build Point to Pixel",
    description="Run ./build_point_to_pixel",
    command=_build_point_to_pixel_command,
    icon='build.png'
)