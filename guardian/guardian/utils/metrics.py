"""
Metrics Utilities
=================
System metrics collection and custom probe interface.
"""

import abc
import dataclasses
import threading
import time
from typing import Dict, List, Optional, Tuple

import psutil


@dataclasses.dataclass
class MetricSample:
    """A single metric data point."""
    name: str
    value: float
    unit: str = ''
    tags: Dict[str, str] = dataclasses.field(default_factory=dict)
    timestamp: float = dataclasses.field(default_factory=time.monotonic)


class MetricProbe(abc.ABC):
    """
    Interface for custom metric probes.

    Implement this to add robot-specific metrics (battery, motor temp, etc.)
    and register via the `probe_plugins` parameter.
    """

    probe_name: str = 'custom_probe'

    @abc.abstractmethod
    def collect(self) -> Optional[MetricSample]:
        """Collect and return a metric sample, or None if unavailable."""


class SystemMetricsCollector:
    """
    Collects system-level and ROS2 metrics for monitored nodes.

    Maintains a per-node sliding window of metrics and exposes
    aggregated statistics.
    """

    def __init__(self, ros_node, history_window: float = 60.0) -> None:
        self._node = ros_node
        self._window = history_window
        self._pid_cache: Dict[str, Optional[int]] = {}
        self._latency_history: Dict[str, List[Tuple[float, float]]] = {}
        self._topic_rate_trackers: Dict[str, Dict[str, '_RateTracker']] = {}
        self._lock = threading.Lock()

    def get_node_pid(self, node_name: str) -> Optional[int]:
        """Get PID for a ROS2 node by matching against running processes."""
        if node_name in self._pid_cache:
            return self._pid_cache[node_name]

        # Search by process cmdline for ros2 run / component containers
        short_name = node_name.strip('/').split('/')[-1]
        for proc in psutil.process_iter(['pid', 'cmdline', 'name']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if short_name in cmdline and 'python' in (proc.info['name'] or '').lower():
                    pid = proc.info['pid']
                    with self._lock:
                        self._pid_cache[node_name] = pid
                    return pid
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        with self._lock:
            self._pid_cache[node_name] = None
        return None

    def is_node_alive(self, node_name: str) -> bool:
        """Check if a node is currently registered with the ROS2 graph."""
        known = [
            f"/{ns}/{name}".replace('//', '/')
            for name, ns in self._node.get_node_names_and_namespaces()
        ]
        return any(node_name in kn for kn in known)

    def get_lifecycle_state(self, node_name: str) -> str:
        """Query lifecycle state for managed nodes."""
        try:
            from lifecycle_msgs.srv import GetState
            client = self._node.create_client(
                GetState, f"{node_name}/get_state"
            )
            if not client.service_is_ready():
                return 'unknown'
            # Synchronous check with short timeout
            req = GetState.Request()
            future = client.call_async(req)
            # Non-blocking: return 'unknown' if not immediately available
            if future.done():
                result = future.result()
                return result.current_state.label if result else 'unknown'
            return 'unknown'
        except Exception:  # noqa: BLE001
            return 'unknown'

    def get_topic_rates(self, node_name: str) -> Dict[str, float]:
        """Return topic → publish rate (Hz) for a node."""
        with self._lock:
            trackers = self._topic_rate_trackers.get(node_name, {})
            return {topic: t.rate_hz for topic, t in trackers.items()}

    def get_avg_latency(self, node_name: str) -> float:
        """Return average message latency in ms over the sliding window."""
        now = time.monotonic()
        with self._lock:
            hist = self._latency_history.get(node_name, [])
            recent = [lat for ts, lat in hist if now - ts <= self._window]
            return sum(recent) / len(recent) if recent else 0.0

    def get_max_latency(self, node_name: str) -> float:
        """Return max message latency in ms over the sliding window."""
        now = time.monotonic()
        with self._lock:
            hist = self._latency_history.get(node_name, [])
            recent = [lat for ts, lat in hist if now - ts <= self._window]
            return max(recent) if recent else 0.0

    def record_latency(self, node_name: str, latency_ms: float) -> None:
        with self._lock:
            if node_name not in self._latency_history:
                self._latency_history[node_name] = []
            now = time.monotonic()
            self._latency_history[node_name].append((now, latency_ms))
            # Trim old entries
            cutoff = now - self._window
            self._latency_history[node_name] = [
                (ts, lat) for ts, lat in self._latency_history[node_name]
                if ts >= cutoff
            ]


class _RateTracker:
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
            recent = [t for t in self._timestamps if now - t <= self.window]
            return len(recent) / self.window if recent else 0.0
