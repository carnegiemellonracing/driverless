FROM ros:foxy

# python 3.9 installation
RUN apt-get update && apt-get install -y \
  python3.9 \
  python3-pip \
  python3-pytest-cov \
  ros-dev-tools \
  wget \
  vim \
  git \
  bash-completion \
  build-essential \
  sudo \
  xauth \
  ros-foxy-rviz2 \
  && rm -rf /var/lib/apt/lists/*

RUN /bin/bash -c "source /opt/ros/foxy/setup.bash"
RUN echo "source /opt/ros/foxy/setup.bash" >> /root/.bashrc
RUN echo "export PYTHONPATH="/root/workspace/cmrdv_ws/src:$PYTHONPATH"" >> /root/.bashrc

WORKDIR /root/workspace/

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV HOME=/root \
  DEBIAN_FRONTEND=noninteractive \
  LANG=en_US.UTF-8 \
  LANGUAGE=en_US.UTF-8 \
  LC_ALL=C.UTF-8 \
  DISPLAY=:0.0 \
  DISPLAY_WIDTH=1024 \
  DISPLAY_HEIGHT=768 \
  RUN_XTERM=yes \
  RUN_FLUXBOX=yes \
  ROS_HOSTNAME=ros
