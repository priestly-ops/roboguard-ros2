"""
Failure Predictor Node
======================
Consumes NodeHealth metrics and SystemAlert messages, runs an LSTM-based
time-series model, and publishes a failure probability score for each node.

When the predicted probability exceeds configured thresholds, a SystemAlert
with increased severity is published to trigger preventive healing.

Architecture:
  NodeHealth  ──►  LSTM Predictor  ──►  risk_score in [0, 1]
  SystemAlert ──►  (context boost)       │
                                         ├── < alert_threshold  → skip
                                         ├── alert_threshold    → WARN alert
                                         └── critical_threshold → CRITICAL alert
"""

import collections
import threading
from typing import Dict, Deque
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor

from guardian_msgs.msg import NodeHealth, SystemAlert
from guardian.ml.failure_predictor import FailurePredictorModel, LSTMPredictor, XGBoostPredictor
from guardian.ml.feature_extractor import FeatureExtractor
from guardian.utils.logger import GuardianLogger


_MODEL_REGISTRY = {
    'lstm': LSTMPredictor,
    'xgboost': XGBoostPredictor,
}


class FailurePredictorNode(Node):
    """
    LSTM-based failure probability predictor.

    Parameters
    ----------
    model_type : str
        One of: lstm, xgboost
    model_path : str
        Path to a pre-trained model checkpoint.
    sequence_length : int
        LSTM input sequence length (number of time steps).
    prediction_horizon_seconds : float
        How far ahead to predict (reported in alert message).
    alert_threshold : float
        Risk score [0,1] that triggers a WARN alert.
    critical_threshold : float
        Risk score that triggers a CRITICAL alert + immediate healing.
    score_publish_rate_hz : float
        Rate at which risk scores are re-evaluated per node.
    """

    NODE_NAME = 'guardian_failure_predictor'

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)
        self._declare_params()

        self.log = GuardianLogger(self.get_logger())
        self.model_type: str = self.get_parameter('model_type').value
        self.model_path: str = self.get_parameter('model_path').value
        self.seq_len: int = self.get_parameter('sequence_length').value
        self.horizon: float = self.get_parameter('prediction_horizon_seconds').value
        self.alert_thresh: float = self.get_parameter('alert_threshold').value
        self.critical_thresh: float = self.get_parameter('critical_threshold').value
        pub_rate: float = self.get_parameter('score_publish_rate_hz').value

        self.log.info(
            f"FailurePredictorNode starting — model={self.model_type}, "
            f"seq_len={self.seq_len}, horizon={self.horizon}s, "
            f"thresholds=[{self.alert_thresh}, {self.critical_thresh}]"
        )

        # ── ML model ──────────────────────────────────────────────────────────
        self._model: FailurePredictorModel = self._init_model()
        self._extractor = FeatureExtractor()
        self._model_lock = threading.Lock()
        self._fitted = bool(self.model_path)

        # ── State ─────────────────────────────────────────────────────────────
        self._sequences: Dict[str, Deque[NodeHealth]] = {}
        self._risk_scores: Dict[str, float] = {}
        self._alert_boost: Dict[str, float] = {}    # Extra risk from recent anomalies
        self._cooldown_tracker: Dict[str, float] = {}

        # ── ROS2 I/O ──────────────────────────────────────────────────────────
        self._metrics_sub = self.create_subscription(
            NodeHealth, '/guardian/metrics', self._on_metrics, 20
        )
        self._alerts_sub = self.create_subscription(
            SystemAlert, '/guardian/alerts', self._on_alert, 10
        )
        self._alert_pub = self.create_publisher(
            SystemAlert, '/guardian/alerts', 10
        )
        self._score_pub = self.create_publisher(
            SystemAlert, '/guardian/risk_scores', 10
        )

        # ── Periodic scoring timer ─────────────────────────────────────────────
        self._timer = self.create_timer(1.0 / max(pub_rate, 0.1), self._publish_scores)

        self.log.info("FailurePredictorNode ready")

    # ──────────────────────────────────────────────────────────────────────────
    # Parameter declarations
    # ──────────────────────────────────────────────────────────────────────────

    def _declare_params(self) -> None:
        self.declare_parameter(
            'model_type', 'lstm',
            ParameterDescriptor(description='Model: lstm | xgboost')
        )
        self.declare_parameter(
            'model_path', '',
            ParameterDescriptor(description='Path to pre-trained model checkpoint')
        )
        self.declare_parameter(
            'sequence_length', 50,
            ParameterDescriptor(description='Input sequence length (time steps)')
        )
        self.declare_parameter(
            'prediction_horizon_seconds', 30.0,
            ParameterDescriptor(description='Prediction lookahead in seconds')
        )
        self.declare_parameter(
            'alert_threshold', 0.75,
            ParameterDescriptor(description='Risk score for WARN alert')
        )
        self.declare_parameter(
            'critical_threshold', 0.90,
            ParameterDescriptor(description='Risk score for CRITICAL alert')
        )
        self.declare_parameter(
            'score_publish_rate_hz', 1.0,
            ParameterDescriptor(description='Rate to re-evaluate and publish risk scores')
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Subscription callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _on_metrics(self, msg: NodeHealth) -> None:
        name = msg.node_name
        if name not in self._sequences:
            self._sequences[name] = collections.deque(maxlen=self.seq_len)
        self._sequences[name].append(msg)

        # Predict when we have a full sequence
        if len(self._sequences[name]) == self.seq_len and self._fitted:
            self._predict(name)

    def _on_alert(self, msg: SystemAlert) -> None:
        """Boost risk score for a node when an anomaly is already detected."""
        name = msg.node_name
        boost_amount = {
            'INFO': 0.05,
            'WARN': 0.15,
            'ERROR': 0.25,
            'CRITICAL': 0.40,
        }.get(msg.severity, 0.10)
        current = self._alert_boost.get(name, 0.0)
        self._alert_boost[name] = min(current + boost_amount, 0.5)

    # ──────────────────────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────────────────────

    def _predict(self, node_name: str) -> None:
        seq = list(self._sequences[node_name])
        features = self._extractor.extract_sequence(seq)
        if features is None:
            return

        with self._model_lock:
            raw_score = self._model.predict_proba(features)

        # Apply alert boost (blended)
        boost = self._alert_boost.get(node_name, 0.0)
        score = min(raw_score + boost, 1.0)

        self._risk_scores[node_name] = score

        # Decay boost over time
        if boost > 0:
            self._alert_boost[node_name] = max(boost - 0.01, 0.0)

    def _publish_scores(self) -> None:
        now = self.get_clock().now()
        import time
        now_mono = time.monotonic()

        for node_name, score in list(self._risk_scores.items()):
            # Publish all scores for the dashboard
            score_msg = SystemAlert()
            score_msg.header.stamp = now.to_msg()
            score_msg.node_name = node_name
            score_msg.anomaly_score = score
            score_msg.severity = 'SCORE'
            score_msg.detector = f'predictor/{self.model_type}'
            self._score_pub.publish(score_msg)

            # Check if alert needed
            if score >= self.alert_thresh:
                cooldown_key = node_name
                last_alert = self._cooldown_tracker.get(cooldown_key, 0.0)
                if now_mono - last_alert < 30.0:
                    continue

                self._cooldown_tracker[cooldown_key] = now_mono

                severity = 'CRITICAL' if score >= self.critical_thresh else 'WARN'
                alert = SystemAlert()
                alert.header.stamp = now.to_msg()
                alert.node_name = node_name
                alert.anomaly_score = score
                alert.severity = severity
                alert.detector = f'predictor/{self.model_type}'
                alert.message = (
                    f"Predicted failure risk for {node_name}: "
                    f"{score:.1%} (within ~{self.horizon:.0f}s)"
                )
                alert.timestamp = now.to_msg()

                self._alert_pub.publish(alert)
                self.log.warn(alert.message)

    # ──────────────────────────────────────────────────────────────────────────
    # Model initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def _init_model(self) -> FailurePredictorModel:
        cls = _MODEL_REGISTRY.get(self.model_type, LSTMPredictor)
        model = cls(sequence_length=self.seq_len)

        if self.model_path:
            try:
                model.load(self.model_path)
                self.log.info(f"Loaded predictor model from {self.model_path}")
            except Exception as exc:  # noqa: BLE001
                self.log.error(f"Failed to load predictor: {exc}")

        return model


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = FailurePredictorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
