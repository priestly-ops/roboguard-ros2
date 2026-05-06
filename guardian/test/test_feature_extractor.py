"""
Tests for guardian.ml.feature_extractor
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from guardian.ml.feature_extractor import FeatureExtractor


def make_msg(cpu=30.0, mem=200.0):
    return {
        'cpu_percent': cpu,
        'memory_mb': mem,
        'memory_percent': 25.0,
        'thread_count': 10,
        'open_files': 5,
        'avg_latency_ms': 2.0,
        'max_latency_ms': 10.0,
        'topic_rates_hz': [10.0, 5.0, 20.0],
        'is_alive': True,
        'lifecycle_state': 'active',
    }


class TestFeatureExtractor:

    def test_extract_returns_correct_shape(self):
        ext = FeatureExtractor()
        window = [make_msg() for _ in range(30)]
        features = ext.extract(window)
        assert features is not None
        assert features.shape == (ext.n_features,)

    def test_extract_short_window_returns_none(self):
        ext = FeatureExtractor()
        window = [make_msg()]
        features = ext.extract(window)
        assert features is None

    def test_extract_empty_window_returns_none(self):
        ext = FeatureExtractor()
        assert ext.extract([]) is None

    def test_extract_sequence_shape(self):
        ext = FeatureExtractor()
        window = [make_msg() for _ in range(50)]
        seq = ext.extract_sequence(window)
        assert seq is not None
        assert seq.shape == (50, FeatureExtractor.N_BASE)

    def test_extract_is_deterministic(self):
        ext = FeatureExtractor()
        window = [make_msg(cpu=float(i)) for i in range(10)]
        f1 = ext.extract(window)
        f2 = ext.extract(window)
        assert np.allclose(f1, f2)

    def test_lifecycle_encoding(self):
        ext = FeatureExtractor()
        msg_active = make_msg()
        msg_active['lifecycle_state'] = 'active'
        msg_error = make_msg()
        msg_error['lifecycle_state'] = 'error'

        feat_active = ext._from_dict(msg_active)
        feat_error = ext._from_dict(msg_error)

        assert feat_active[-1] == 3.0   # active = 3
        assert feat_error[-1] == -1.0   # error = -1

    def test_n_features_matches_extract(self):
        ext = FeatureExtractor()
        window = [make_msg() for _ in range(5)]
        features = ext.extract(window)
        assert features.shape[0] == ext.n_features
