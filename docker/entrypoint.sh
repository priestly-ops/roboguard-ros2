#!/bin/bash
# Guardian Docker entrypoint
set -e

# Source ROS2 and workspace
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

echo "🤖 ROS2 Guardian — Starting"
echo "   ROS_DOMAIN_ID: ${ROS_DOMAIN_ID:-0}"
echo "   Workspace: /ros2_ws"
echo "   Time: $(date -u)"

exec "$@"
