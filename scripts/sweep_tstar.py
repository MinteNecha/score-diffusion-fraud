#!/usr/bin/env python3
"""Sweep the noise level t* and see how it affects the precision-recall
trade-off of the diffusion anomaly detector.

This is the investigation flagged in the proposal as not covered by the
anchor paper (Livernoche et al., ICLR 2024): the diffusion model is
trained once, then evaluated at many different t* values at test time
(no retraining needed - t* only affects how the trained model is *used*
for scoring). A small t* means only light noise is added, so the model
barely has to "fix" anything and errors stay small for everyone -
possibly too subtle to separate fraud from normal. A large t* means
almost all the original signal is destroyed by noise, so the model is
reconstructing from very little information and errors get large and
noisy for everyone too. Somewhere in between should be the sweet spot.

Usage:
    python scripts/sweep_tstar.py --dataset creditcard
    python scripts/sweep_tstar.py --dataset creditcard --t-values 5 10 20 30 50 70 90
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.data.loaders import get_dataset  # noqa: E402
from src.evaluation.metrics import evaluate_all  # noqa: E402
from src.training.train_diffusion import train_diffusion_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="creditcard",
        choices=["creditcard", "bank_account_fraud", "paysim"],
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--t-values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 30, 40, 50, 60, 70, 85, 99],
        help="Noise levels t* to evaluate at (must be < diffusion.timesteps in the config).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, overrides={"dataset": {"name": args.dataset}})
    set_seed(cfg.seed)

    print(f"=== Loading dataset: {args.dataset} ===")
    data = get_dataset(
        args.dataset,
        raw_dir=PROJECT_ROOT / cfg.dataset.raw_dir,
        test_size=cfg.dataset.test_size,
        val_size=cfg.dataset.val_size,
        seed=cfg.seed,
    )

    print("=== Training diffusion model (once) ===")
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

    rows = []
    print("\n=== Sweeping t* ===")
    for t_star in args.t_values:
        if t_star >= cfg.diffusion.timesteps:
            print(f"  skipping t*={t_star} (>= timesteps={cfg.diffusion.timesteps})")
            continue
        scores = diffusion.anomaly_score(X_test_t, t_star=t_star, n_repeats=5).numpy()
        metrics = evaluate_all(data.y_test, scores, target_fpr=cfg.evaluation.target_fpr)
        metrics["t_star"] = t_star
        rows.append(metrics)
        print(f"  t*={t_star:>4d}  {metrics}")

    df = pd.DataFrame(rows)
    cols = ["t_star", "auroc", "average_precision"] + [c for c in df.columns if c.startswith("f1_at_")]
    df = df[cols]

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / f"tstar_sweep_{args.dataset}.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== t* sweep results ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {out_csv}")

    best_row = df.loc[df["auroc"].idxmax()]
    print(f"\nBest AUROC at t*={int(best_row['t_star'])} (AUROC={best_row['auroc']:.4f})")


if __name__ == "__main__":
    main()
