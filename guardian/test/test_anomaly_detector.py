"""
Tests for guardian.ml.anomaly_detector
"""

import numpy as np
import pytest

# Use plain Python for offline testing (no ROS2 needed)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from guardian.ml.anomaly_detector import (
    IsolationForestDetector,
    OneClassSVMDetector,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def make_normal_data(n=500, n_features=14, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.5, scale=0.1, size=(n, n_features)).astype(np.float32)


def make_anomalous_data(n=20, n_features=14, seed=99):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=2.0, scale=0.5, size=(n, n_features)).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# IsolationForestDetector
# ──────────────────────────────────────────────────────────────────────────────

class TestIsolationForestDetector:

    def test_fit_and_score_normal(self):
        model = IsolationForestDetector(contamination=0.05)
        X_train = make_normal_data()
        model.fit(X_train)

        X_normal = make_normal_data(n=10)
        for sample in X_normal:
            score = model.score(sample.reshape(1, -1))
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1]"
            assert score < 0.7, f"Normal sample scored too high: {score}"

    def test_fit_and_score_anomalous(self):
        model = IsolationForestDetector(contamination=0.05)
        X_train = make_normal_data()
        model.fit(X_train)

        X_anom = make_anomalous_data(n=10)
        scores = [model.score(s.reshape(1, -1)) for s in X_anom]
        avg_score = np.mean(scores)
        # On average, anomalies should score higher than normals
        assert avg_score > 0.4, f"Anomalies should score higher, got avg={avg_score}"

    def test_score_before_fit_returns_zero(self):
        model = IsolationForestDetector()
        X = make_normal_data(n=1)
        score = model.score(X)
        assert score == 0.0

    def test_save_and_load(self, tmp_path):
        model = IsolationForestDetector()
        X_train = make_normal_data()
        model.fit(X_train)

        path = str(tmp_path / "model.pkl")
        model.save(path)

        model2 = IsolationForestDetector()
        model2.load(path)

        X = make_normal_data(n=5)
        for sample in X:
            s1 = model.score(sample.reshape(1, -1))
            s2 = model2.score(sample.reshape(1, -1))
            assert abs(s1 - s2) < 1e-5, "Loaded model gives different scores"

    def test_fit_with_small_dataset(self):
        """Should not raise with very small training set."""
        model = IsolationForestDetector()
        X = make_normal_data(n=20)
        model.fit(X)
        score = model.score(X[:1])
        assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# OneClassSVMDetector
# ──────────────────────────────────────────────────────────────────────────────

class TestOneClassSVMDetector:

    def test_fit_and_score(self):
        model = OneClassSVMDetector(contamination=0.05)
        X_train = make_normal_data(n=200)
        model.fit(X_train)

        X_normal = make_normal_data(n=5)
        for sample in X_normal:
            score = model.score(sample.reshape(1, -1))
            assert 0.0 <= score <= 1.0

    def test_score_before_fit_returns_zero(self):
        model = OneClassSVMDetector()
        score = model.score(make_normal_data(n=1))
        assert score == 0.0
