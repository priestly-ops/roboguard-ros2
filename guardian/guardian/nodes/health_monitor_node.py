"""
Health Monitor Node
===================
Collects per-node health metrics across the entire ROS2 system.
Publishes NodeHealth messages on /guardian/metrics at a configurable rate.

Monitored metrics:
  - CPU usage (%), memory usage (MB / %)
  - Topic publish rate vs. expected rate
  - Message latency (last stamp delta)
  - Node lifecycle state
  - TF tree freshness
  - Custom user-defined probes
"""

import time
import psutil
import threading
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import Header

from guardian_msgs.msg import NodeHealth, MetricSample
from guardian.utils.metrics import MetricProbe, SystemMetricsCollector
from guardian.utils.logger import GuardianLogger


class HealthMonitorNode(Node):
    """
    Continuously monitors all running ROS2 nodes and publishes health metrics.

    Parameters
    ----------
    monitor_rate_hz : float
        Publishing rate for health metrics (default: 10.0)
    history_window_seconds : float
        Sliding window of metric history kept in memory (default: 60.0)
    watched_nodes : list[str]
        Explicit node list to monitor. Empty = monitor all nodes.
    probe_plugins : list[str]
        Dotted-path to custom MetricProbe subclasses to load.
    """

    NODE_NAME = 'guardian_health_monitor'

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)

        # ── Parameters ────────────────────────────────────────────────────────
        self._declare_params()
        self.monitor_rate = self.get_parameter('monitor_rate_hz').value
        self.history_window = self.get_parameter('history_window_seconds').value
        self.watched_nodes: List[str] = self.get_parameter('watched_nodes').value or []

        self.log = GuardianLogger(self.get_logger())
        self.log.info(
            f"HealthMonitorNode starting — rate={self.monitor_rate} Hz, "
            f"window={self.history_window}s"
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self._health_pub = self.create_publisher(
            NodeHealth, '/guardian/metrics', qos_profile=10
        )

        # ── Internal state ────────────────────────────────────────────────────
        self._collector = SystemMetricsCollector(self, self.history_window)
        self._metric_history: Dict[str, List[MetricSample]] = {}
        self._topic_rate_tracker: Dict[str, _TopicRateTracker] = {}
        self._lock = threading.Lock()

        # ── Load custom probes ────────────────────────────────────────────────
        self._probes: List[MetricProbe] = self._load_probes()

        # ── Timer ─────────────────────────────────────────────────────────────
        period = 1.0 / max(self.monitor_rate, 0.1)
        self._timer = self.create_timer(period, self._on_timer)

        self.log.info("HealthMonitorNode ready")

    # ──────────────────────────────────────────────────────────────────────────
    # ROS2 lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def _declare_params(self) -> None:
        self.declare_parameter(
            'monitor_rate_hz', 10.0,
            ParameterDescriptor(description='Metric publishing rate in Hz')
        )
        self.declare_parameter(
            'history_window_seconds', 60.0,
            ParameterDescriptor(description='Sliding metric history window in seconds')
        )
        self.declare_parameter(
            'watched_nodes', [],
            ParameterDescriptor(description='Nodes to monitor. Empty = all nodes.')
        )
        self.declare_parameter(
            'probe_plugins', [],
            ParameterDescriptor(description='Dotted paths to custom MetricProbe subclasses')
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Timer callback
    # ──────────────────────────────────────────────────────────────────────────

    def _on_timer(self) -> None:
        try:
            nodes = self._get_target_nodes()
            for node_name in nodes:
                msg = self._build_health_msg(node_name)
                self._health_pub.publish(msg)
        except Exception as exc:  # noqa: BLE001
            self.log.error(f"Health monitor tick failed: {exc}")

    # ──────────────────────────────────────────────────────────────────────────
    # Metric collection
    # ──────────────────────────────────────────────────────────────────────────

    def _get_target_nodes(self) -> List[str]:
        if self.watched_nodes:
            return self.watched_nodes
        # Discover all nodes dynamically (minus ourselves)
        all_nodes = [
            f"/{ns}/{name}".replace('//', '/')
            for name, ns in self.get_node_names_and_namespaces()
        ]
        return [n for n in all_nodes if self.NODE_NAME not in n]

    def _build_health_msg(self, node_name: str) -> NodeHealth:
        now = self.get_clock().now().to_msg()
        pid = self._collector.get_node_pid(node_name)

        msg = NodeHealth()
        msg.header = Header(stamp=now, frame_id='guardian')
        msg.node_name = node_name
        msg.timestamp = now
        msg.is_alive = self._collector.is_node_alive(node_name)
        msg.lifecycle_state = self._collector.get_lifecycle_state(node_name)

        # Process-level metrics (if PID is available)
        if pid is not None:
            try:
                proc = psutil.Process(pid)
                msg.cpu_percent = proc.cpu_percent(interval=None)
                mem = proc.memory_info()
                msg.memory_mb = mem.rss / (1024 * 1024)
                msg.memory_percent = proc.memory_percent()
                msg.thread_count = proc.num_threads()
                msg.open_files = len(proc.open_files())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Topic throughput
        topic_rates = self._collector.get_topic_rates(node_name)
        msg.published_topics = list(topic_rates.keys())
        msg.topic_rates_hz = list(topic_rates.values())

        # Latency
        msg.avg_latency_ms = self._collector.get_avg_latency(node_name)
        msg.max_latency_ms = self._collector.get_max_latency(node_name)

        # Custom probes
        for probe in self._probes:
            try:
                sample = probe.collect()
                if sample:
                    msg.custom_metrics.append(sample)
            except Exception as exc:  # noqa: BLE001
                self.log.warn(f"Probe {probe.probe_name} failed: {exc}")

        return msg

    def _load_probes(self) -> List[MetricProbe]:
        plugin_paths: List[str] = self.get_parameter('probe_plugins').value or []
        probes: List[MetricProbe] = []
        for path in plugin_paths:
            try:
                module_path, class_name = path.rsplit('.', 1)
                import importlib
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                probes.append(cls())
                self.log.info(f"Loaded probe plugin: {path}")
            except Exception as exc:  # noqa: BLE001
                self.log.error(f"Failed to load probe {path}: {exc}")
        return probes


class _TopicRateTracker:
    """Tracks message rate for a single topic over a sliding window."""

    def __init__(self, window: float = 5.0) -> None:
        self.window = window
        self._timestamps: List[float] = []
        self._lock = threading.Lock()

    def record(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._timestamps.append(now)
            cutoff = now - self.window
            self._timestamps = [t for t in self._timestamps if t >= cutoff]

    @property
    def rate_hz(self) -> float:
        now = time.monotonic()
        with self._lock:
            cutoff = now - self.window
            recent = [t for t in self._timestamps if t >= cutoff]
            return len(recent) / self.window if recent else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = HealthMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
