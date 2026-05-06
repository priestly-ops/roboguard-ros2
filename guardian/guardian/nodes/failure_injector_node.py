#!/usr/bin/env python3
"""Failure injector for Guardian simulation.

Injects configurable faults into mock robot nodes at runtime so Guardian's
anomaly detection and self-healing logic can be exercised end-to-end.
"""

import json
import random
import time
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Parameter as RclParameter
from rcl_interfaces.msg import ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from std_msgs.msg import String


FAILURE_MODES = ["cpu_spike", "memory_leak", "topic_drop", "crash", "none"]


class FailureInjectorNode(Node):
    """Randomly or deterministically injects failures into mock robot nodes."""

    def __init__(self):
        super().__init__("failure_injector")

        self.declare_parameter("target_nodes", ["mock_navigation", "mock_slam", "mock_arm"])
        self.declare_parameter("injection_interval_sec", 60.0)
        self.declare_parameter("random_mode", True)
        self.declare_parameter("scenario", "random")  # random|sequential|burst
        self.declare_parameter("dry_run", False)

        self._targets = self.get_parameter("target_nodes").value
        self._interval = self.get_parameter("injection_interval_sec").value
        self._random = self.get_parameter("random_mode").value
        self._scenario = self.get_parameter("scenario").value
        self._dry_run = self.get_parameter("dry_run").value

        self._event_pub = self.create_publisher(String, "/guardian/sim/injection_events", 10)
        self._set_param_clients: dict[str, rclpy.client.Client] = {}

        self._injection_index = 0
        self._history: list[dict] = []

        self._timer = self.create_timer(self._interval, self._inject)
        self.get_logger().info(
            f"FailureInjector ready — targets={self._targets}, "
            f"scenario={self._scenario}, interval={self._interval}s"
        )

    # ------------------------------------------------------------------
    def _inject(self):
        if self._scenario == "sequential":
            target = self._targets[self._injection_index % len(self._targets)]
            mode = FAILURE_MODES[self._injection_index % len(FAILURE_MODES)]
        elif self._scenario == "burst":
            target = random.choice(self._targets)
            mode = random.choice(FAILURE_MODES[:-1])  # never inject 'none'
        else:  # random
            target = random.choice(self._targets)
            mode = random.choice(FAILURE_MODES)

        severity = round(random.uniform(0.5, 1.0), 2)
        self._injection_index += 1

        event = {
            "ts": time.time(),
            "target": target,
            "mode": mode,
            "severity": severity,
            "dry_run": self._dry_run,
        }
        self._history.append(event)

        msg = String()
        msg.data = json.dumps(event)
        self._event_pub.publish(msg)

        self.get_logger().warn(
            f"[INJECT] target={target} mode={mode} severity={severity}"
            + (" (DRY RUN)" if self._dry_run else "")
        )

        if not self._dry_run:
            self._set_failure_param(target, mode, severity)

    # ------------------------------------------------------------------
    def _set_failure_param(self, node_name: str, mode: str, severity: float):
        if node_name not in self._set_param_clients:
            self._set_param_clients[node_name] = self.create_client(
                SetParameters, f"/{node_name}/set_parameters"
            )

        client = self._set_param_clients[node_name]
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f"SetParameters service unavailable for {node_name}")
            return

        req = SetParameters.Request()

        def _str_param(name: str, val: str) -> RclParameter:
            p = RclParameter()
            p.name = name
            p.value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=val)
            return p

        def _float_param(name: str, val: float) -> RclParameter:
            p = RclParameter()
            p.name = name
            p.value = ParameterValue(
                type=ParameterType.PARAMETER_DOUBLE, double_value=val
            )
            return p

        req.parameters = [
            _str_param("failure_mode", mode),
            _float_param("failure_start_sec", 0.0),
            _float_param("failure_severity", severity),
        ]

        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if future.done():
            self.get_logger().info(f"Injected {mode} into {node_name}")
        else:
            self.get_logger().warn(f"Injection timed out for {node_name}")


def main(args=None):
    rclpy.init(args=args)
    node = FailureInjectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
