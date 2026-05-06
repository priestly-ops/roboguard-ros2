"""
Anomaly Detector Node
=====================
Subscribes to /guardian/metrics, feeds metric windows into an ML model
(Isolation Forest by default), and publishes SystemAlert messages when
anomalous patterns are detected.

Detection pipeline:
  1. Buffer incoming NodeHealth messages into per-node sliding windows
  2. Extract feature vectors from each window
  3. Score features with the loaded model
  4. Threshold scores → SystemAlert with severity level
  5. Optionally retrain model on-line with accumulated data

Supported models:  isolation_forest | autoencoder | one_class_svm
"""

import time
import threading
import collections
from typing import Dict, List, Optional, Deque

import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor

from guardian_msgs.msg import NodeHealth, SystemAlert
from guardian.ml.anomaly_detector import (
    AnomalyDetectorModel,
    IsolationForestDetector,
    AutoencoderDetector,
    OneClassSVMDetector,
)
from guardian.ml.feature_extractor import FeatureExtractor
from guardian.utils.logger import GuardianLogger


_MODEL_REGISTRY = {
    'isolation_forest': IsolationForestDetector,
    'autoencoder': AutoencoderDetector,
    'one_class_svm': OneClassSVMDetector,
}

_SEVERITY_LEVELS = {
    0.0: 'INFO',
    0.5: 'WARN',
    0.75: 'ERROR',
    0.90: 'CRITICAL',
}


class AnomalyDetectorNode(Node):
    """
    Detects anomalous node behaviour using an ML model on metric windows.

    Parameters
    ----------
    model_type : str
        One of: isolation_forest, autoencoder, one_class_svm
    model_path : str
        Path to a pre-trained model file (.pkl / .pt). Empty = train from scratch.
    window_size : int
        Number of consecutive NodeHealth messages per feature window.
    contamination : float
        Expected fraction of anomalous data (used by Isolation Forest).
    alert_cooldown_seconds : float
        Minimum time between consecutive alerts for the same node.
    retrain_interval_minutes : float
        How often to retrain the model on accumulated data. 0 = disabled.
    """

    NODE_NAME = 'guardian_anomaly_detector'
    WINDOW_SIZE_DEFAULT = 30

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)
        self._declare_params()

        self.log = GuardianLogger(self.get_logger())
        self.model_type: str = self.get_parameter('model_type').value
        self.model_path: str = self.get_parameter('model_path').value
        self.window_size: int = self.get_parameter('window_size').value
        self.contamination: float = self.get_parameter('contamination').value
        self.cooldown: float = self.get_parameter('alert_cooldown_seconds').value
        self.retrain_interval: float = self.get_parameter('retrain_interval_minutes').value * 60

        self.log.info(
            f"AnomalyDetectorNode starting — model={self.model_type}, "
            f"window={self.window_size}, contamination={self.contamination}"
        )

        # ── ML model ──────────────────────────────────────────────────────────
        self._model: AnomalyDetectorModel = self._init_model()
        self._feature_extractor = FeatureExtractor()
        self._model_lock = threading.Lock()
        self._fitted = False

        # ── Per-node metric buffers ────────────────────────────────────────────
        self._buffers: Dict[str, Deque[NodeHealth]] = {}
        self._training_data: List[np.ndarray] = []
        self._alert_timestamps: Dict[str, float] = {}

        # ── ROS2 I/O ──────────────────────────────────────────────────────────
        self._metrics_sub = self.create_subscription(
            NodeHealth, '/guardian/metrics', self._on_metrics, 20
        )
        self._alert_pub = self.create_publisher(
            SystemAlert, '/guardian/alerts', 10
        )

        # ── Retrain timer ──────────────────────────────────────────────────────
        if self.retrain_interval > 0:
            self._retrain_timer = self.create_timer(
                self.retrain_interval, self._retrain_model
            )

        self.log.info("AnomalyDetectorNode ready")

    # ──────────────────────────────────────────────────────────────────────────
    # Parameter declarations
    # ──────────────────────────────────────────────────────────────────────────

    def _declare_params(self) -> None:
        self.declare_parameter(
            'model_type', 'isolation_forest',
            ParameterDescriptor(description='ML model: isolation_forest | autoencoder | one_class_svm')
        )
        self.declare_parameter(
            'model_path', '',
            ParameterDescriptor(description='Path to pre-trained model file')
        )
        self.declare_parameter(
            'window_size', self.WINDOW_SIZE_DEFAULT,
            ParameterDescriptor(description='Feature window size in number of samples')
        )
        self.declare_parameter(
            'contamination', 0.05,
            ParameterDescriptor(description='Expected anomaly fraction for Isolation Forest')
        )
        self.declare_parameter(
            'alert_cooldown_seconds', 30.0,
            ParameterDescriptor(description='Minimum seconds between alerts for same node')
        )
        self.declare_parameter(
            'retrain_interval_minutes', 30.0,
            ParameterDescriptor(description='Retrain interval in minutes. 0 = disabled.')
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Subscription callback
    # ──────────────────────────────────────────────────────────────────────────

    def _on_metrics(self, msg: NodeHealth) -> None:
        node_name = msg.node_name

        # Maintain per-node sliding window
        if node_name not in self._buffers:
            self._buffers[node_name] = collections.deque(maxlen=self.window_size)
        self._buffers[node_name].append(msg)

        window = self._buffers[node_name]

        # Accumulate training data (pre-fit)
        features = self._feature_extractor.extract(list(window))
        if features is not None:
            self._training_data.append(features)

        # Fit model on first sufficient batch
        if not self._fitted and len(self._training_data) >= self.window_size * 5:
            self._fit_model()

        # Detect anomalies once fitted
        if self._fitted and len(window) == self.window_size:
            self._detect(node_name, features)

    # ──────────────────────────────────────────────────────────────────────────
    # Detection
    # ──────────────────────────────────────────────────────────────────────────

    def _detect(self, node_name: str, features: Optional[np.ndarray]) -> None:
        if features is None:
            return

        with self._model_lock:
            score = self._model.score(features.reshape(1, -1))

        # score is in [0, 1] — higher = more anomalous
        if score < 0.5:
            return

        # Cooldown check
        now = time.monotonic()
        last = self._alert_timestamps.get(node_name, 0.0)
        if now - last < self.cooldown:
            return

        self._alert_timestamps[node_name] = now
        severity = self._score_to_severity(score)

        alert = SystemAlert()
        alert.header.stamp = self.get_clock().now().to_msg()
        alert.node_name = node_name
        alert.anomaly_score = float(score)
        alert.severity = severity
        alert.message = (
            f"Anomaly detected on {node_name} "
            f"(score={score:.3f}, severity={severity})"
        )
        alert.detector = self.model_type
        alert.timestamp = self.get_clock().now().to_msg()

        self._alert_pub.publish(alert)
        self.log.warn(alert.message)

    @staticmethod
    def _score_to_severity(score: float) -> str:
        level = 'INFO'
        for threshold, sev in sorted(_SEVERITY_LEVELS.items()):
            if score >= threshold:
                level = sev
        return level

    # ──────────────────────────────────────────────────────────────────────────
    # Model management
    # ──────────────────────────────────────────────────────────────────────────

    def _init_model(self) -> AnomalyDetectorModel:
        cls = _MODEL_REGISTRY.get(self.model_type)
        if cls is None:
            self.log.warn(
                f"Unknown model type '{self.model_type}', falling back to isolation_forest"
            )
            cls = IsolationForestDetector

        model = cls(contamination=self.contamination)

        if self.model_path:
            try:
                model.load(self.model_path)
                self._fitted = True
                self.log.info(f"Loaded pre-trained model from {self.model_path}")
            except Exception as exc:  # noqa: BLE001
                self.log.error(f"Failed to load model: {exc}. Will train from scratch.")

        return model

    def _fit_model(self) -> None:
        X = np.vstack(self._training_data)
        with self._model_lock:
            self._model.fit(X)
            self._fitted = True
        self.log.info(
            f"Model fitted on {len(X)} samples (shape={X.shape})"
        )

    def _retrain_model(self) -> None:
        if len(self._training_data) < 50:
            return
        self.log.info("Retraining anomaly model with accumulated data…")
        self._fit_model()
        # Optionally trim old data
        self._training_data = self._training_data[-5000:]


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = AnomalyDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
