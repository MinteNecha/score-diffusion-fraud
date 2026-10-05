"""Train diffusion, autoencoder and Random Forest once on one dataset, then
save precision-recall curves, score histograms and a scoring-speed report.

Usage:
    python scripts/make_figures.py --dataset creditcard --t-star 94
    python scripts/make_figures.py --dataset bank_account_fraud --t-star 1
    python scripts/make_figures.py --dataset paysim --t-star 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from src.data.loaders import get_dataset  # noqa: E402
from src.models.baselines import AutoencoderBaseline, RandomForestBaseline  # noqa: E402
from src.training.train_diffusion import train_diffusion_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

N_REPEATS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["creditcard", "bank_account_fraud", "paysim"])
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--t-star", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    overrides = {"dataset": {"name": args.dataset}, "seed": args.seed}
    if args.t_star is not None:
        overrides["diffusion"] = {"t_star": args.t_star}
    cfg = load_config(args.config, overrides=overrides)
    set_seed(cfg.seed)
    t_star = cfg.diffusion.t_star

    data = get_dataset(
        args.dataset,
        raw_dir=PROJECT_ROOT / cfg.dataset.raw_dir,
        test_size=cfg.dataset.test_size,
        val_size=cfg.dataset.val_size,
        seed=cfg.seed,
    )
    y = data.y_test

    print("Training diffusion model...")
    diffusion = train_diffusion_model(
        data.X_train_normal,
        X_val_normal=data.X_val_normal,
        hidden_dim=cfg.diffusion.hidden_dim,
        n_hidden_layers=cfg.diffusion.n_hidden_layers,
        time_embed_dim=cfg.diffusion.time_embed_dim,
        timesteps=cfg.diffusion.timesteps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        lr=cfg.diffusion.lr,
        batch_size=cfg.diffusion.batch_size,
        epochs=cfg.diffusion.epochs,
        weight_decay=cfg.diffusion.weight_decay,
        seed=cfg.seed,
    )
    X_test_t = torch.tensor(data.X_test, dtype=torch.float32)
    t0 = time.time()
    diff_scores = diffusion.anomaly_score(X_test_t, t_star=t_star, n_repeats=N_REPEATS).numpy()
    diff_seconds = time.time() - t0

    print("Training autoencoder...")
    ae = AutoencoderBaseline(
        in_dim=data.X_train_normal.shape[1],
        hidden_dim=cfg.baselines.autoencoder.hidden_dim,
        latent_dim=cfg.baselines.autoencoder.latent_dim,
        lr=cfg.baselines.autoencoder.lr,
        batch_size=cfg.baselines.autoencoder.batch_size,
        epochs=cfg.baselines.autoencoder.epochs,
        seed=cfg.seed,
    ).fit(data.X_train_normal)
    t0 = time.time()
    ae_scores = ae.anomaly_score(data.X_test)
    ae_seconds = time.time() - t0

    print("Training Random Forest...")
    rf = RandomForestBaseline(
        n_estimators=cfg.baselines.random_forest.n_estimators,
        max_depth=cfg.baselines.random_forest.max_depth,
        seed=cfg.seed,
    ).fit(data.X_train_full, data.y_train_full)
    t0 = time.time()
    rf_scores = rf.anomaly_score(data.X_test)
    rf_seconds = time.time() - t0

    out_dir = PROJECT_ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"scores_{args.dataset}.npz", y=y, diffusion=diff_scores, autoencoder=ae_scores, random_forest=rf_scores)

    models = {
        f"Diffusion (t*={t_star})": diff_scores,
        "Autoencoder": ae_scores,
        "Random Forest": rf_scores,
    }

    fig, ax = plt.subplots(figsize=(5, 4))
    for name, s in models.items():
        precision, recall, _ = precision_recall_curve(y, s)
        ax.plot(recall, precision, label=f"{name} (AP={average_precision_score(y, s):.3f})")
    ax.axhline(y.mean(), color="grey", linestyle=":", label="Random (fraud rate)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall curves: {args.dataset}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / f"pr_{args.dataset}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    upper = np.percentile(diff_scores, 99.5)
    bins = np.linspace(diff_scores.min(), upper, 60)
    ax.hist(diff_scores[y == 0], bins=bins, alpha=0.6, density=True, label="Normal")
    ax.hist(diff_scores[y == 1], bins=bins, alpha=0.6, density=True, label="Fraud")
    ax.set_xlabel("Diffusion anomaly score")
    ax.set_ylabel("Density")
    ax.set_title(f"Score distribution: {args.dataset}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"scores_hist_{args.dataset}.png", dpi=200)
    plt.close(fig)

    n = len(y)
    report = (
        f"Scoring time on {n} test transactions ({args.dataset}):\n"
        f"  diffusion (t*={t_star}, {N_REPEATS} repeats): {diff_seconds:.2f}s = {diff_seconds / n * 1e6:.1f} microseconds per transaction\n"
        f"  autoencoder: {ae_seconds:.2f}s = {ae_seconds / n * 1e6:.1f} microseconds per transaction\n"
        f"  random forest: {rf_seconds:.2f}s = {rf_seconds / n * 1e6:.1f} microseconds per transaction\n"
    )
    print(report)
    (out_dir / f"latency_{args.dataset}.txt").write_text(report)
    print(f"Saved figures and scores to {out_dir}")


if __name__ == "__main__":
    main()