# Perceptions ROS Package

Dale requires restart!

Software Versions Required
- `gcc` (Version 9.4.0+)
- `g++` (Version 9.4.0+)
- `nvcc` (Version 12.3.107+)
    - Install from [here](https://developer.nvidia.com/cuda-downloads)
- NVIDIA Driver (Version 545+)
    - Install drivers from [here](https://www.nvidia.com/download/index.aspx)

Packages Required
- `eufs_msgs`
- `HesaiLidar_ROS_2.0` (follow [this](https://github.com/carnegiemellonracing/HesaiLidar_ROS_2.0) for build instructions)
- `ros2_numpy`
- `vision_opencv`
- `zed-ros2-wrapper` (follow [this](https://github.com/carnegiemellonracing/zed-ros2-wrapper) for build instructions)
    - Before above, install ZED SDK from [here](https://www.stereolabs.com/developers/release).
    - If fails, use `sudo apt install zstd`