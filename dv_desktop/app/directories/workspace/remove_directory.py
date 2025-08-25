from utils.types import Module
import subprocess

def _remove_directory_command():
    subprocess.run(["rm", "-r", "build"], cwd = "../driverless_ws")
    subprocess.run(["rm", "-r", "log"], cwd = "../driverless_ws")
    subprocess.run(["rm", "-r", "install"], cwd = "../driverless_ws")


RemoveDirectoryModule = Module(
    title="Remove Directories",
    description="Remove build, log, install directories from driverless_ws",
    command=_remove_directory_command,
    icon='trash-2.png'
)