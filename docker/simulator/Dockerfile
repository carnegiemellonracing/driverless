FROM osrf/ros:galactic-desktop

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
RUN apt-get install -y ros-galactic-gazebo-ros \
                       ros-galactic-gazebo-ros-pkgs \
                       ros-galactic-xacro \
                       ros-galactic-ackermann-msgs \
                       ros-galactic-joint-state-publisher-gui

RUN apt-get upgrade -y ros-galactic-geometry-msgs

RUN echo 'export EUFS_MASTER=/eufs' >> ~/.bashrc
RUN echo 'source /opt/ros/galactic/setup.bash' >> ~/.bashrc
RUN source ~/.bashrc

RUN mkdir eufs
RUN cd eufs && \
    git clone https://gitlab.com/eufs/eufs_sim.git && \
    git clone https://gitlab.com/eufs/eufs_msgs.git && \
    git clone https://gitlab.com/eufs/eufs_rviz_plugins.git

RUN cd eufs && rosdep update && rosdep install --from-paths /eufs --ignore-src -r -y
 

