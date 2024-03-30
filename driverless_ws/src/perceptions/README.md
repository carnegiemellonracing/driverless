# Perceptions ROS Package

Software Versions Required
- `gcc` (Version 9.4.0+)
- `g++` (Version 9.4.0+)
- `nvcc` (Version 12.3.107+)
    - Check by running `nvcc --version`
    - Install from [here](https://developer.nvidia.com/cuda-downloads)
    - Ensure environment variables `$PATH` and `$LD_LIBRARY_PATH` contain appropriate binary and library paths
- NVIDIA Driver (Version 545+)
    - Check by running `nvidia-smi`
    - Install drivers from [here](https://www.nvidia.com/download/index.aspx)
    - Requires computer restart at times

Packages Required
- `HesaiLidar_ROS_2.0` (follow [this](https://github.com/carnegiemellonracing/HesaiLidar_ROS_2.0) for build instructions)
    - Note that `hesai_ros_driver` must be built with `--symlink-install`
- `ros2_numpy`
- `vision_opencv`
- `zed-ros2-wrapper` (follow [this](https://github.com/carnegiemellonracing/zed-ros2-wrapper) for build instructions)
    - Before installing `zed-ros2-wrapper`, install ZED SDK as listed below
        1. Run `sudo apt install zstd`
        2. Install ZED SDK from [here](https://www.stereolabs.com/developers/release).

Currently, for ZED data, the `ZEDNode` which publishes raw sensor data does not publish a right image or a depth image
because they aren't used in the pipeline.

Additionally, because we aren't currently running the `zed-ros2-wrapper` code, avoid attempts to build
`zed_components` due to lack of dependencies (which `rosdep` doesn't install for some reason).
