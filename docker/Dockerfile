# ROS2 Guardian — Multi-stage Dockerfile
# =========================================
# Stage 1: builder  — installs Python deps and builds ROS2 packages
# Stage 2: runtime  — lean image for deployment

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM ros:humble-ros-base AS builder

LABEL maintainer="guardian@your-org.com"
LABEL description="ROS2 Guardian — AI-Powered Predictive Failure Detection"

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV WORKSPACE=/ros2_ws

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-setuptools \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python ML dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Set up workspace
WORKDIR ${WORKSPACE}/src
COPY guardian_msgs/ ./guardian_msgs/
COPY guardian/ ./guardian/

WORKDIR ${WORKSPACE}
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    rosdep init || true && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build --packages-select guardian_msgs guardian \
      --cmake-args -DCMAKE_BUILD_TYPE=Release"


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM ros:humble-ros-base AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV WORKSPACE=/ros2_ws

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    psmisc \
    && rm -rf /var/lib/apt/lists/*

# Copy built workspace from builder
COPY --from=builder ${WORKSPACE}/install ${WORKSPACE}/install
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Copy config
COPY config/ ${WORKSPACE}/config/

# Entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8765    # WebSocket dashboard bridge

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "guardian", "guardian.launch.py"]
