"""Training loop for the score-based diffusion anomaly detector.

Trains the DDPM noise predictor on normal-only transactions, with a
simple validation loss tracked on a held-out normal split for basic
sanity checking (there are no labels to validate against at this stage -
that happens later, at evaluation time, via anomaly_score + AUROC/AP/F1).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from src.models.diffusion import GaussianDiffusion, MLPDenoiser

logger = logging.getLogger(__name__)


def train_diffusion_model(
    X_train_normal: np.ndarray,
    X_val_normal: Optional[np.ndarray] = None,
    hidden_dim: int = 128,
    n_hidden_layers: int = 3,
    time_embed_dim: int = 32,
    timesteps: int = 100,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    lr: float = 1e-3,
    batch_size: int = 256,
    epochs: int = 50,
    weight_decay: float = 0.0,
    device: str = "cpu",
    seed: int = 42,
    log_every: int = 10,
) -> GaussianDiffusion:
    """Train an MLPDenoiser wrapped in a GaussianDiffusion process.

    Returns the fitted GaussianDiffusion object, ready for
    `.anomaly_score(x, t_star)` calls at evaluation time.
    """
    torch.manual_seed(seed)

    data_dim = X_train_normal.shape[1]
    model = MLPDenoiser(
        data_dim=data_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        time_embed_dim=time_embed_dim,
    )
    diffusion = GaussianDiffusion(
        model,
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
    )

    optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=lr, weight_decay=weight_decay)

    X_train_t = torch.tensor(X_train_normal, dtype=torch.float32, device=device)
    X_val_t = (
        torch.tensor(X_val_normal, dtype=torch.float32, device=device)
        if X_val_normal is not None
        else None
    )
    n = X_train_t.shape[0]

    for epoch in range(1, epochs + 1):
        diffusion.model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train_t[idx]

            loss = diffusion.training_loss(xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        if epoch % log_every == 0 or epoch == epochs:
            msg = f"[diffusion] epoch {epoch}/{epochs}  train_loss={avg_train_loss:.5f}"
            if X_val_t is not None:
                diffusion.model.eval()
                with torch.no_grad():
                    val_loss = diffusion.training_loss(X_val_t).item()
                msg += f"  val_loss={val_loss:.5f}"
            logger.info(msg)
            print(msg)

    return diffusion
