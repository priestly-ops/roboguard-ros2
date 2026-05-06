#!/usr/bin/env python3
"""Mock robot node for Guardian simulation.

Simulates a realistic robot subsystem (navigation, SLAM, arm controller, etc.)
publishing synthetic metrics that Guardian can monitor.
"""

import random
import time
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MockRobotNode(Node):
    """Simulates a robot subsystem with configurable failure modes."""

    def __init__(self):
        super().__init__("mock_robot_node")

        # Parameters
        self.declare_parameter("node_role", "navigation")
        self.declare_parameter("publish_hz", 10.0)
        self.declare_parameter("base_cpu", 15.0)
        self.declare_parameter("base_memory_mb", 256.0)
        self.declare_parameter("failure_mode", "none")   # none|cpu_spike|memory_leak|topic_drop|crash
        self.declare_parameter("failure_start_sec", 30.0)
        self.declare_parameter("failure_severity", 0.8)

        self.role = self.get_parameter("node_role").value
        self.hz = self.get_parameter("publish_hz").value
        self.base_cpu = self.get_parameter("base_cpu").value
        self.base_mem = self.get_parameter("base_memory_mb").value
        self.failure_mode = self.get_parameter("failure_mode").value
        self.failure_start = self.get_parameter("failure_start_sec").value
        self.failure_severity = self.get_parameter("failure_severity").value

        self._start_time = time.time()
        self._step = 0

        # Role-specific topic publishers
        self._role_pubs = {}
        if self.role == "navigation":
            self._role_pubs["cmd_vel"] = self.create_publisher(String, "/cmd_vel", 10)
            self._role_pubs["path"] = self.create_publisher(String, "/plan", 10)
        elif self.role == "slam":
            self._role_pubs["map"] = self.create_publisher(String, "/map", 10)
            self._role_pubs["pose"] = self.create_publisher(String, "/slam/pose", 10)
        elif self.role == "arm":
            self._role_pubs["joint_states"] = self.create_publisher(String, "/joint_states", 10)
            self._role_pubs["trajectory"] = self.create_publisher(String, "/arm/trajectory", 10)
        elif self.role == "sensor":
            self._role_pubs["scan"] = self.create_publisher(String, "/scan", 10)
            self._role_pubs["imu"] = self.create_publisher(String, "/imu/data", 10)

        self._status_pub = self.create_publisher(String, f"/mock/{self.role}/status", 10)

        period = 1.0 / self.hz
        self._timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f"MockRobotNode started: role={self.role}, failure_mode={self.failure_mode}"
        )

    # ------------------------------------------------------------------
    def _tick(self):
        elapsed = time.time() - self._start_time
        self._step += 1

        # Compute realistic metrics
        cpu, mem = self._compute_metrics(elapsed)

        # Determine if we should drop topics (topic_drop failure)
        drop_topics = False
        if self.failure_mode == "topic_drop" and elapsed > self.failure_start:
            drop_factor = min(1.0, (elapsed - self.failure_start) / 10.0)
            drop_topics = random.random() < drop_factor * self.failure_severity

        # Crash simulation — node raises an exception
        if self.failure_mode == "crash" and elapsed > self.failure_start:
            self.get_logger().error(f"[{self.role}] Simulated crash triggered!")
            raise RuntimeError(f"Simulated crash in {self.role} node")

        if not drop_topics:
            msg = String()
            msg.data = (
                f'{{"role":"{self.role}","step":{self._step},'
                f'"cpu":{cpu:.1f},"mem":{mem:.1f},"elapsed":{elapsed:.1f}}}'
            )
            for pub in self._role_pubs.values():
                pub.publish(msg)
            self._status_pub.publish(msg)

    # ------------------------------------------------------------------
    def _compute_metrics(self, elapsed: float):
        """Return (cpu_pct, mem_mb) with optional failure injection."""
        noise = random.gauss(0, 2.0)
        cpu = self.base_cpu + noise + 5 * math.sin(elapsed / 30.0)
        mem = self.base_mem + random.gauss(0, 5.0)

        if elapsed < self.failure_start:
            return max(0.0, cpu), max(0.0, mem)

        t = elapsed - self.failure_start

        if self.failure_mode == "cpu_spike":
            spike = min(80.0, t * 4.0) * self.failure_severity
            cpu += spike + random.gauss(0, spike * 0.1)

        elif self.failure_mode == "memory_leak":
            leak = min(2048.0, t * 10.0) * self.failure_severity
            mem += leak

        return max(0.0, min(100.0, cpu)), max(0.0, mem)


def main(args=None):
    rclpy.init(args=args)
    node = MockRobotNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError) as e:
        node.get_logger().warn(f"Node stopped: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
