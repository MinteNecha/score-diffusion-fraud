"""Baseline models to compare against the diffusion anomaly detector.

Per the proposal (topic_a.pdf), four baselines are used:
  - RandomForestBaseline   : supervised, needs fraud labels
  - FeedforwardNNBaseline  : supervised, needs fraud labels
  - IsolationForestBaseline: unsupervised, no labels (same setting as diffusion)
  - AutoencoderBaseline    : unsupervised reconstruction-error detector,
                              the closest "classic" analogue to the
                              diffusion model's own reconstruction-error idea

Every baseline exposes a common two-method interface:
  - fit(X, y=None)               : y is ignored / unused for unsupervised models
  - anomaly_score(X) -> np.ndarray : higher score = more anomalous / more likely fraud
so `scripts/run_experiment.py` can loop over all of them identically.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest, RandomForestClassifier


class RandomForestBaseline:
    """Supervised classifier. anomaly_score = predicted probability of fraud."""

    def __init__(self, n_estimators: int = 200, max_depth: Optional[int] = None, seed: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        self.model.fit(X, y)
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class _SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_hidden_layers: int, out_dim: int):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FeedforwardNNBaseline:
    """Supervised binary classifier trained with BCE loss."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        lr: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 30,
        device: str = "cpu",
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = _SimpleMLP(in_dim, hidden_dim, n_hidden_layers, out_dim=1).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeedforwardNNBaseline":
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        n = X_t.shape[0]

        # class imbalance is severe (fraud << normal); weight the positive
        # class so the network doesn't just predict "normal" for everything
        pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], device=self.device)

        self.model.train()
        for _epoch in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb, yb = X_t[idx], y_t[idx]
                logits = self.model(xb).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self

    @torch.no_grad()
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        logits = self.model(X_t).squeeze(-1)
        return torch.sigmoid(logits).cpu().numpy()


class IsolationForestBaseline:
    """Unsupervised. Trained on normal-only data, same setting as the diffusion model."""

    def __init__(self, n_estimators: int = 200, contamination="auto", seed: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators, contamination=contamination, random_state=seed, n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "IsolationForestBaseline":
        # y is ignored - Isolation Forest is unsupervised. Fit on normal-only X.
        self.model.fit(X)
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        # sklearn's score_samples is higher = more normal, so flip the sign
        return -self.model.score_samples(X)


class _AEModule(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class AutoencoderBaseline:
    """Unsupervised reconstruction-error detector - the classic analogue to
    the diffusion model's own reconstruction-error idea, trained on normal
    data only. anomaly_score = ||x - decoder(encoder(x))||_2.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        lr: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 30,
        device: str = "cpu",
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = _AEModule(in_dim, hidden_dim, latent_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AutoencoderBaseline":
        # y is ignored - trained on normal-only X, same as the diffusion model.
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        n = X_t.shape[0]

        self.model.train()
        for _epoch in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb = X_t[idx]
                recon = self.model(xb)
                loss = F.mse_loss(recon, xb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self

    @torch.no_grad()
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        recon = self.model(X_t)
        err = torch.linalg.vector_norm(X_t - recon, dim=-1)
        return err.cpu().numpy()
