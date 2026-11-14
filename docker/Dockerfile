FROM osrf/ros:humble-desktop

# UFactory xarm_ros2 (humble branch) with all submodules
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        python3-colcon-common-extensions \
        python3-pip \
        python3-vcstool \
        ros-humble-moveit \
        ros-humble-realsense2-camera \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /ws/src \
    && git clone --recursive -b humble https://github.com/xArm-Developer/xarm_ros2.git /ws/src/xarm_ros2 \
    && rosdep update \
    && rosdep install --from-paths /ws/src --ignore-src -r -y \
    && . /opt/ros/humble/setup.sh \
    && cd /ws \
    && colcon build --symlink-install

COPY requirements.lock /tmp/requirements.lock

# Hash-pinned wheels only; then a non-root user (uid 1000 matches the usual host user so mounted files stay writable)
RUN pip3 install --no-cache-dir --only-binary :all: --require-hashes -r /tmp/requirements.lock \
    && useradd --create-home --uid 1000 dev \
    && chown -R dev:dev /ws \
    && echo 'source /opt/ros/humble/setup.bash; source /ws/install/setup.bash 2>/dev/null' >> /home/dev/.bashrc

USER dev
WORKDIR /project
