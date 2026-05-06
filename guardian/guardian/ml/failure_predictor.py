"""
Failure Predictor Models
========================
Time-series models that predict the probability of node failure
within the next N seconds.

Available:
  - LSTMPredictor    (PyTorch, best for temporal patterns)
  - XGBoostPredictor (gradient boosting on windowed features, fast)
"""

import abc
import pickle
from typing import Optional

import numpy as np


class FailurePredictorModel(abc.ABC):
    """Abstract base for failure prediction models."""

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on sequence features X and binary labels y (1 = pre-failure)."""

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> float:
        """Return failure probability in [0, 1] for input X."""

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Persist model."""

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """Restore model."""


# ──────────────────────────────────────────────────────────────────────────────
# LSTM Predictor
# ──────────────────────────────────────────────────────────────────────────────

class LSTMPredictor(FailurePredictorModel):
    """
    Bidirectional LSTM classifier for failure prediction.

    Architecture:
      Input (seq_len, n_features)
        └─ BiLSTM (hidden=128, layers=2, dropout=0.2)
           └─ Attention pooling
              └─ FC(128 → 64 → 1)
                 └─ Sigmoid → probability

    Parameters
    ----------
    sequence_length : int
        Number of time-steps per input window.
    input_dim : int
        Feature dimensionality (inferred on first fit if 0).
    hidden_dim : int
        LSTM hidden size.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout rate.
    epochs : int
        Training epochs.
    lr : float
        Adam learning rate.
    """

    def __init__(
        self,
        sequence_length: int = 50,
        input_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> None:
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self._model = None

    def _build(self, input_dim: int):
        try:
            import torch
            import torch.nn as nn

            class _BiLSTMClassifier(nn.Module):
                def __init__(self, in_d, h_d, n_layers, drop):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        in_d, h_d, num_layers=n_layers,
                        dropout=drop, batch_first=True, bidirectional=True
                    )
                    self.attention = nn.Linear(h_d * 2, 1)
                    self.fc = nn.Sequential(
                        nn.Linear(h_d * 2, 64),
                        nn.ReLU(),
                        nn.Dropout(drop),
                        nn.Linear(64, 1),
                        nn.Sigmoid(),
                    )

                def forward(self, x):
                    out, _ = self.lstm(x)           # (B, T, 2H)
                    # Attention
                    weights = torch.softmax(self.attention(out), dim=1)  # (B, T, 1)
                    context = (out * weights).sum(dim=1)                  # (B, 2H)
                    return self.fc(context).squeeze(-1)

            return _BiLSTMClassifier(input_dim, self.hidden_dim, self.num_layers, self.dropout)
        except ImportError:
            return None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        X: shape (N, sequence_length, n_features)
        y: shape (N,) — binary labels
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            self.input_dim = X.shape[2]
            self._model = self._build(self.input_dim)
            if self._model is None:
                return

            X_t = torch.FloatTensor(X)
            y_t = torch.FloatTensor(y)

            # Class-weight balanced BCE loss
            pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)]).float()
            loss_fn = nn.BCELoss()

            loader = DataLoader(
                TensorDataset(X_t, y_t), batch_size=32, shuffle=True
            )
            opt = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

            self._model.train()
            for epoch in range(self.epochs):
                for xb, yb in loader:
                    opt.zero_grad()
                    pred = self._model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                    opt.step()
                scheduler.step()

            self._model.eval()

        except ImportError:
            pass

    def predict_proba(self, X: np.ndarray) -> float:
        if self._model is None:
            return 0.0
        try:
            import torch

            self._model.eval()
            with torch.no_grad():
                # X shape: (seq_len, features) → (1, seq_len, features)
                if X.ndim == 2:
                    X = X[np.newaxis, ...]
                x_t = torch.FloatTensor(X)
                prob = self._model(x_t).item()
            return float(max(0.0, min(1.0, prob)))
        except Exception:  # noqa: BLE001
            return 0.0

    def save(self, path: str) -> None:
        try:
            import torch
            torch.save({
                'state_dict': self._model.state_dict() if self._model else None,
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'sequence_length': self.sequence_length,
            }, path)
        except ImportError:
            pass

    def load(self, path: str) -> None:
        try:
            import torch
            data = torch.load(path, map_location='cpu')
            self.input_dim = data['input_dim']
            self.hidden_dim = data['hidden_dim']
            self.num_layers = data['num_layers']
            self.dropout = data['dropout']
            self.sequence_length = data['sequence_length']
            if data['state_dict']:
                self._model = self._build(self.input_dim)
                self._model.load_state_dict(data['state_dict'])
                self._model.eval()
        except ImportError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# XGBoost Predictor
# ──────────────────────────────────────────────────────────────────────────────

class XGBoostPredictor(FailurePredictorModel):
    """
    XGBoost classifier on flattened + windowed feature vectors.

    Faster than LSTM, good for deployments without GPU.
    Features are flattened (seq_len * n_features) with added
    rolling statistics (mean, std, min, max per feature).

    Parameters
    ----------
    sequence_length : int
        Input sequence length.
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Tree max depth.
    """

    def __init__(
        self,
        sequence_length: int = 50,
        n_estimators: int = 300,
        max_depth: int = 6,
        **kwargs,
    ) -> None:
        self.sequence_length = sequence_length
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._model = None

    def _engineer_features(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, seq_len, features) → (N, augmented_features)
        """
        flat = X.reshape(X.shape[0], -1)
        mean = X.mean(axis=1)
        std = X.std(axis=1)
        mn = X.min(axis=1)
        mx = X.max(axis=1)
        # Trend (last - first) per feature
        trend = X[:, -1, :] - X[:, 0, :]
        return np.hstack([flat, mean, std, mn, mx, trend])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            import xgboost as xgb

            X_feat = self._engineer_features(X)
            scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
            self._model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1,
                random_state=42,
            )
            self._model.fit(X_feat, y)
        except ImportError:
            # Fallback to sklearn GBM
            from sklearn.ensemble import GradientBoostingClassifier
            X_feat = self._engineer_features(X)
            self._model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
            )
            self._model.fit(X_feat, y)

    def predict_proba(self, X: np.ndarray) -> float:
        if self._model is None:
            return 0.0
        if X.ndim == 2:
            X = X[np.newaxis, ...]
        X_feat = self._engineer_features(X)
        proba = self._model.predict_proba(X_feat)[0]
        return float(proba[1]) if len(proba) > 1 else 0.0

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self._model, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            self._model = pickle.load(f)
