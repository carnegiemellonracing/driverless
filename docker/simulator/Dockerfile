FROM ubuntu:20.04
ENV DEBIAN_FRONTEND noninteractive
SHELL ["/bin/bash", "-c"]
ADD start_simulator.txt /
RUN apt-get update
RUN apt-get install -y \
    python3-pip \
    tmux \
    vim \
    nano
RUN pip3 install colcon-common-extensions -U
RUN apt-get install -y python3-tk
RUN apt-get install -y software-properties-common
RUN add-apt-repository universe
RUN apt-get update && apt-get install -y curl
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y ros-foxy-desktop
RUN apt-get install -y ros-foxy-gazebo-ros \
                       ros-foxy-gazebo-ros-pkgs \
                       ros-foxy-xacro \
                       ros-foxy-ackermann-msgs \
                       ros-foxy-joint-state-publisher-gui
RUN apt-get upgrade -y ros-foxy-geometry-msgs
RUN echo 'export EUFS_MASTER=/eufs' >> ~/.bashrc
RUN echo 'source /opt/ros/foxy/setup.bash' >> ~/.bashrc
RUN source ~/.bashrc
RUN apt-get -y install git
RUN apt-get install -y python3-rosdep
RUN rosdep init
RUN mkdir eufs
RUN cd eufs && \
    git clone https://gitlab.com/eufs/eufs_sim.git && \
    git clone https://gitlab.com/eufs/eufs_msgs.git && \
    git clone https://gitlab.com/eufs/eufs_rviz_plugins.git
RUN cd eufs && rosdep update && rosdep install --from-paths /eufs --ignore-src -r -y
