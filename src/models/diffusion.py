"""Score-based / DDPM-style diffusion model for tabular fraud data.

Follows the anchor paper (Livernoche et al., "On Diffusion Modeling for
Anomaly Detection", ICLR 2024): a standard DDPM is trained ONLY on normal
transactions. At test time, a transaction is noised to a fixed level t*,
denoised by the model, and the reconstruction error between the original
and the denoised version is used as the anomaly score. Normal transactions
sit on the manifold the model learned and are easy to restore (low error);
fraud does not, so the model "corrects" it more heavily (high error).

Two pieces:
  - sinusoidal_embedding : turns an integer timestep into a vector, the
    same trick used for positional encodings in Transformers.
  - MLPDenoiser           : the actual noise-prediction network, a plain
    MLP since the data here is tabular (rows of numbers, not images).
  - GaussianDiffusion     : wraps the denoiser with the forward noising
    process (q_sample), the training objective (training_loss), and the
    test-time anomaly scoring logic (denoise_at_t / anomaly_score).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Map integer timesteps (B,) to sinusoidal embeddings (B, dim).

    Identical idea to Transformer positional encodings: alternating sin/cos
    at geometrically spaced frequencies, so the network can tell nearby
    timesteps apart while still generalising smoothly across t.
    """
    device = timesteps.device
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=device, dtype=torch.float32) / half_dim
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:  # zero-pad if dim is odd
        embedding = F.pad(embedding, (0, 1))
    return embedding


class MLPDenoiser(nn.Module):
    """Predicts the noise added to a (noised) input vector at timestep t.

    Input:  x_t (B, data_dim), t (B,)
    Output: predicted noise eps_hat (B, data_dim)
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 128,
        n_hidden_layers: int = 3,
        time_embed_dim: int = 32,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        layers = [nn.Linear(data_dim + time_embed_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, data_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(t, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)
        h = torch.cat([x_t, t_emb], dim=-1)
        return self.net(h)


class GaussianDiffusion:
    """Standard DDPM forward/reverse process wrapped around an MLPDenoiser.

    Uses a linear beta schedule from beta_start to beta_end over
    `timesteps` steps, exactly as in Ho et al. (2020).
    """

    def __init__(
        self,
        model: MLPDenoiser,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # ---- forward (noising) process ----
    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_0) in closed form: x_t = sqrt(a_bar_t) x0 + sqrt(1-a_bar_t) eps."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_1m_ac = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        return sqrt_ac * x0 + sqrt_1m_ac * noise

    # ---- training objective ----
    def training_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Simple DDPM loss: MSE between the true noise and the predicted noise,
        with t sampled uniformly at random for each example in the batch.
        """
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device).long()
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_t, t)
        return F.mse_loss(noise_pred, noise)

    # ---- test-time denoising / anomaly scoring ----
    @torch.no_grad()
    def denoise_at_t(self, x0: torch.Tensor, t_star: int, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Noise x0 to level t_star, then take a single denoising step back
        to an estimate of x0 (the "one-shot" reconstruction used by the
        anchor paper's anomaly score, cheaper than running the full reverse
        chain since we only need a reconstruction error, not a sample).

        x0_hat = (x_t - sqrt(1 - a_bar_t) * eps_hat) / sqrt(a_bar_t)
        """
        self.model.eval()
        b = x0.shape[0]
        t = torch.full((b,), t_star, device=x0.device, dtype=torch.long)
        x_t = self.q_sample(x0, t, noise)
        eps_hat = self.model(x_t, t)

        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_1m_ac = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        x0_hat = (x_t - sqrt_1m_ac * eps_hat) / sqrt_ac
        return x0_hat

    @torch.no_grad()
    def anomaly_score(
        self, x0: torch.Tensor, t_star: int, n_repeats: int = 1
    ) -> torch.Tensor:
        """Reconstruction-error anomaly score at a fixed noise level t*.

        score(x) = || x - denoise_at_t(x, t*) ||_2

        `n_repeats` averages the score over multiple random noise draws to
        reduce variance from the stochastic forward step (the anchor paper
        notes this matters more at higher t*, where a single noise sample
        can be unlucky).
        """
        self.model.eval()
        scores = []
        for _ in range(n_repeats):
            x0_hat = self.denoise_at_t(x0, t_star)
            err = torch.linalg.vector_norm(x0 - x0_hat, dim=-1)
            scores.append(err)
        return torch.stack(scores, dim=0).mean(dim=0)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
