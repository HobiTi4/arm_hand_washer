FROM osrf/ros:humble-desktop
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-colcon-common-extensions python3-vcstool git \
        ros-humble-moveit ros-humble-realsense2-camera \
    && rm -rf /var/lib/apt/lists/*
# UFactory xarm_ros2 (humble branch) з усіма сабмодулями
RUN mkdir -p /ws/src && cd /ws/src \
    && git clone https://github.com/xArm-Developer/xarm_ros2.git --recursive -b humble
RUN cd /ws && rosdep update && rosdep install --from-paths src --ignore-src -r -y \
    && . /opt/ros/humble/setup.sh && colcon build --symlink-install
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN echo 'source /opt/ros/humble/setup.bash; source /ws/install/setup.bash 2>/dev/null' >> /root/.bashrc
WORKDIR /project
