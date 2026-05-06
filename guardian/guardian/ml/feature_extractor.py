"""
Feature Extractor
=================
Converts NodeHealth message sequences into numerical feature vectors
suitable for ML model input.

Feature groups:
  - Resource: cpu_percent, memory_mb, memory_percent, thread_count
  - Throughput: mean topic rate, rate std, min/max rate
  - Latency: avg_latency_ms, max_latency_ms
  - State: is_alive (0/1), lifecycle_state encoded
  - Derived: rate_of_change per metric, rolling statistics
"""

from typing import Optional

import numpy as np


_LIFECYCLE_MAP = {
    '': 0,
    'unknown': 0,
    'unconfigured': 1,
    'inactive': 2,
    'active': 3,
    'finalized': 4,
    'error': -1,
}


class FeatureExtractor:
    """
    Extracts flat feature vectors from a list of NodeHealth messages.

    Designed to work with both real rclpy messages and lightweight
    dictionaries (for testing / offline training).
    """

    # Feature names in extraction order
    BASE_FEATURES = [
        'cpu_percent',
        'memory_mb',
        'memory_percent',
        'thread_count',
        'open_files',
        'avg_latency_ms',
        'max_latency_ms',
        'mean_topic_rate',
        'std_topic_rate',
        'min_topic_rate',
        'max_topic_rate',
        'n_topics',
        'is_alive',
        'lifecycle_state',
    ]

    N_BASE = len(BASE_FEATURES)

    def extract(self, window: list) -> Optional[np.ndarray]:
        """
        Extract a flat feature vector from a window of NodeHealth messages.

        If the window is too short, returns None.
        The vector includes mean, std, min, max, trend of each base feature.

        Returns
        -------
        np.ndarray of shape (N_BASE * 5,) or None
        """
        if not window:
            return None

        samples = np.array([self._extract_one(msg) for msg in window])  # (W, F)
        if samples.shape[0] < 2:
            return None

        mean = samples.mean(axis=0)
        std = samples.std(axis=0) + 1e-8
        mn = samples.min(axis=0)
        mx = samples.max(axis=0)
        trend = samples[-1] - samples[0]

        return np.concatenate([mean, std, mn, mx, trend])

    def extract_sequence(self, window: list) -> Optional[np.ndarray]:
        """
        Extract a 2D sequence (seq_len, n_features) for LSTM input.

        Returns
        -------
        np.ndarray of shape (len(window), N_BASE) or None
        """
        if not window:
            return None
        seq = np.array([self._extract_one(msg) for msg in window])
        return seq

    @property
    def n_features(self) -> int:
        """Number of features in a flat extract() output."""
        return self.N_BASE * 5

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_one(self, msg) -> np.ndarray:
        """Extract base features from a single NodeHealth message."""
        if isinstance(msg, dict):
            return self._from_dict(msg)
        return self._from_msg(msg)

    def _from_msg(self, msg) -> np.ndarray:
        """Extract from a guardian_msgs/NodeHealth ROS2 message."""
        topic_rates = list(getattr(msg, 'topic_rates_hz', []))
        mean_rate = float(np.mean(topic_rates)) if topic_rates else 0.0
        std_rate = float(np.std(topic_rates)) if len(topic_rates) > 1 else 0.0
        min_rate = float(min(topic_rates)) if topic_rates else 0.0
        max_rate = float(max(topic_rates)) if topic_rates else 0.0
        n_topics = float(len(topic_rates))

        lifecycle_raw = getattr(msg, 'lifecycle_state', 'unknown') or 'unknown'
        lifecycle = float(_LIFECYCLE_MAP.get(lifecycle_raw.lower(), 0))

        return np.array([
            float(getattr(msg, 'cpu_percent', 0.0)),
            float(getattr(msg, 'memory_mb', 0.0)),
            float(getattr(msg, 'memory_percent', 0.0)),
            float(getattr(msg, 'thread_count', 0)),
            float(getattr(msg, 'open_files', 0)),
            float(getattr(msg, 'avg_latency_ms', 0.0)),
            float(getattr(msg, 'max_latency_ms', 0.0)),
            mean_rate,
            std_rate,
            min_rate,
            max_rate,
            n_topics,
            float(getattr(msg, 'is_alive', True)),
            lifecycle,
        ], dtype=np.float32)

    def _from_dict(self, d: dict) -> np.ndarray:
        """Extract from a plain dict (for offline use / testing)."""
        topic_rates = d.get('topic_rates_hz', [])
        mean_rate = float(np.mean(topic_rates)) if topic_rates else 0.0
        std_rate = float(np.std(topic_rates)) if len(topic_rates) > 1 else 0.0

        lifecycle_raw = str(d.get('lifecycle_state', 'unknown')).lower()
        lifecycle = float(_LIFECYCLE_MAP.get(lifecycle_raw, 0))

        return np.array([
            float(d.get('cpu_percent', 0.0)),
            float(d.get('memory_mb', 0.0)),
            float(d.get('memory_percent', 0.0)),
            float(d.get('thread_count', 0)),
            float(d.get('open_files', 0)),
            float(d.get('avg_latency_ms', 0.0)),
            float(d.get('max_latency_ms', 0.0)),
            mean_rate,
            std_rate,
            float(min(topic_rates)) if topic_rates else 0.0,
            float(max(topic_rates)) if topic_rates else 0.0,
            float(len(topic_rates)),
            float(d.get('is_alive', True)),
            lifecycle,
        ], dtype=np.float32)
