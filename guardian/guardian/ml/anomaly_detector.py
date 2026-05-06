"""
Anomaly Detector Models
=======================
Pluggable ML models for detecting anomalous node behaviour.
All models implement the AnomalyDetectorModel interface.

Available:
  - IsolationForestDetector   (sklearn, fast, no labels required)
  - AutoencoderDetector        (PyTorch, reconstruction error)
  - OneClassSVMDetector        (sklearn, kernel-based)
"""

import abc
import pickle
from typing import Optional

import numpy as np


class AnomalyDetectorModel(abc.ABC):
    """Abstract base for anomaly detection models."""

    @abc.abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Train the model on normal operating data."""

    @abc.abstractmethod
    def score(self, X: np.ndarray) -> float:
        """
        Return an anomaly score in [0, 1].
        0 = normal, 1 = highly anomalous.
        """

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Persist the model to disk."""

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """Load the model from disk."""


# ──────────────────────────────────────────────────────────────────────────────
# Isolation Forest
# ──────────────────────────────────────────────────────────────────────────────

class IsolationForestDetector(AnomalyDetectorModel):
    """
    Isolation Forest anomaly detector.

    Fast, no labels required, works well on tabular metric data.
    Anomaly score is derived from the negative decision function.

    Parameters
    ----------
    contamination : float
        Expected fraction of anomalies in training data.
    n_estimators : int
        Number of trees.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import RobustScaler

        self.contamination = contamination
        self._scaler = RobustScaler()
        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X: np.ndarray) -> None:
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled)
        self._fitted = True

    def score(self, X: np.ndarray) -> float:
        if not self._fitted:
            return 0.0
        X_scaled = self._scaler.transform(X)
        # decision_function: negative = more anomalous, range varies
        raw = self._model.decision_function(X_scaled)[0]
        # Normalise to [0, 1]: decision_function is roughly [-0.5, 0.5]
        normalised = max(0.0, min(1.0, (-raw + 0.5)))
        return float(normalised)

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump({'scaler': self._scaler, 'model': self._model}, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._scaler = data['scaler']
        self._model = data['model']
        self._fitted = True


# ──────────────────────────────────────────────────────────────────────────────
# Autoencoder
# ──────────────────────────────────────────────────────────────────────────────

class AutoencoderDetector(AnomalyDetectorModel):
    """
    Autoencoder-based anomaly detector (PyTorch).

    Trains a fully-connected autoencoder on normal data.
    High reconstruction error → anomaly.

    Parameters
    ----------
    input_dim : int
        Feature vector dimensionality (inferred on first fit if 0).
    hidden_dim : int
        Encoder hidden layer width.
    latent_dim : int
        Bottleneck dimensionality.
    epochs : int
        Training epochs.
    lr : float
        Adam learning rate.
    contamination : float
        Used to set the anomaly score threshold (percentile).
    """

    def __init__(
        self,
        input_dim: int = 0,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        epochs: int = 50,
        lr: float = 1e-3,
        contamination: float = 0.05,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.contamination = contamination
        self._threshold: float = 0.5
        self._model: Optional[object] = None

    def _build_model(self, input_dim: int):
        try:
            import torch.nn as nn

            class _AE(nn.Module):
                def __init__(self, in_d, h_d, l_d):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(in_d, h_d), nn.ReLU(),
                        nn.Linear(h_d, l_d), nn.ReLU(),
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(l_d, h_d), nn.ReLU(),
                        nn.Linear(h_d, in_d),
                    )

                def forward(self, x):
                    return self.decoder(self.encoder(x))

            return _AE(input_dim, self.hidden_dim, self.latent_dim)
        except ImportError:
            return None

    def fit(self, X: np.ndarray) -> None:
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            self.input_dim = X.shape[1]
            self._model = self._build_model(self.input_dim)
            if self._model is None:
                return

            X_tensor = torch.FloatTensor(X)
            loader = DataLoader(TensorDataset(X_tensor), batch_size=64, shuffle=True)
            optimiser = torch.optim.Adam(self._model.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()

            self._model.train()
            for _ in range(self.epochs):
                for (batch,) in loader:
                    optimiser.zero_grad()
                    recon = self._model(batch)
                    loss = loss_fn(recon, batch)
                    loss.backward()
                    optimiser.step()

            # Set threshold at (1 - contamination) percentile of training errors
            self._model.eval()
            with torch.no_grad():
                recon = self._model(X_tensor)
                errors = ((X_tensor - recon) ** 2).mean(dim=1).numpy()
            self._threshold = float(np.percentile(errors, (1 - self.contamination) * 100))

        except ImportError:
            pass  # Fallback: PyTorch not installed

    def score(self, X: np.ndarray) -> float:
        if self._model is None:
            return 0.0
        try:
            import torch

            self._model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(X)
                recon = self._model(x)
                error = float(((x - recon) ** 2).mean().item())
            return min(1.0, error / (self._threshold + 1e-8))
        except Exception:  # noqa: BLE001
            return 0.0

    def save(self, path: str) -> None:
        try:
            import torch
            torch.save({
                'model_state': self._model.state_dict() if self._model else None,
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'latent_dim': self.latent_dim,
                'threshold': self._threshold,
            }, path)
        except ImportError:
            pass

    def load(self, path: str) -> None:
        try:
            import torch
            data = torch.load(path, map_location='cpu')
            self.input_dim = data['input_dim']
            self.hidden_dim = data['hidden_dim']
            self.latent_dim = data['latent_dim']
            self._threshold = data['threshold']
            if data['model_state']:
                self._model = self._build_model(self.input_dim)
                self._model.load_state_dict(data['model_state'])
                self._model.eval()
        except ImportError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# One-Class SVM
# ──────────────────────────────────────────────────────────────────────────────

class OneClassSVMDetector(AnomalyDetectorModel):
    """
    One-Class SVM anomaly detector (sklearn).

    Uses an RBF kernel.  Slower than Isolation Forest but can capture
    non-linear boundaries.

    Parameters
    ----------
    nu : float
        Upper bound on fraction of training errors (≈ contamination).
    kernel : str
        Kernel type (default: 'rbf').
    contamination : float
        Alias for nu (for API consistency).
    """

    def __init__(
        self,
        nu: float = 0.05,
        kernel: str = 'rbf',
        contamination: float = 0.05,
    ) -> None:
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import RobustScaler

        self._scaler = RobustScaler()
        self._model = OneClassSVM(nu=contamination, kernel=kernel, gamma='scale')
        self._fitted = False

    def fit(self, X: np.ndarray) -> None:
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled)
        self._fitted = True

    def score(self, X: np.ndarray) -> float:
        if not self._fitted:
            return 0.0
        X_scaled = self._scaler.transform(X)
        raw = self._model.decision_function(X_scaled)[0]
        return float(max(0.0, min(1.0, -raw + 0.5)))

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump({'scaler': self._scaler, 'model': self._model}, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._scaler = data['scaler']
        self._model = data['model']
        self._fitted = True
