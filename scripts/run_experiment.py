#!/usr/bin/env python3
"""Run the full comparison: diffusion anomaly detector vs all baselines,
on one dataset, and save a results table.

Usage:
    python scripts/run_experiment.py --dataset creditcard
    python scripts/run_experiment.py --dataset creditcard --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data.loaders import get_dataset  # noqa: E402
from src.evaluation.metrics import evaluate_all  # noqa: E402
from src.models.baselines import (  # noqa: E402
    AutoencoderBaseline,
    FeedforwardNNBaseline,
    IsolationForestBaseline,
    RandomForestBaseline,
)
from src.training.train_diffusion import train_diffusion_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def _load_tuned_rf_params(dataset_name: str) -> dict:
    """Pick up scripts/tune_rf.py's output for this dataset, if it exists."""
    path = PROJECT_ROOT / "results" / f"rf_best_params_{dataset_name}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        params = json.load(f)
    print(f"[random_forest] using tuned params from {path}: {params}")
    return params

def run(dataset_name: str, config_path: str, seed: int | None = None, t_star: int | None = None) -> pd.DataFrame:
    overrides = {"dataset": {"name": dataset_name}}
    if seed is not None:
        overrides["seed"] = seed
    if t_star is not None:
        overrides["diffusion"] = {"t_star": t_star}
    tuned_rf = _load_tuned_rf_params(dataset_name)
    if tuned_rf:
        overrides["baselines"] = {"random_forest": tuned_rf}
    cfg = load_config(config_path, overrides=overrides)

    print(f"\n=== Loading dataset: {dataset_name} ===")
    data = get_dataset(
        dataset_name,
        raw_dir=PROJECT_ROOT / cfg.dataset.raw_dir,
        test_size=cfg.dataset.test_size,
        val_size=cfg.dataset.val_size,
        seed=cfg.seed,
    )
    print(
        f"  train_normal={data.X_train_normal.shape}  val_normal={data.X_val_normal.shape}  "
        f"test={data.X_test.shape}  fraud_rate={data.fraud_rate:.4%}  "
        f"n_features={len(data.feature_names)}"
    )

    results = []

    # --- Diffusion model (unsupervised, normal-only) ---
    print("\n--- Training diffusion model ---")
    t0 = time.time()
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
    train_time = time.time() - t0

    import torch

    X_test_t = torch.tensor(data.X_test, dtype=torch.float32)
    scores = diffusion.anomaly_score(X_test_t, t_star=cfg.diffusion.t_star, n_repeats=5).numpy()
    metrics = evaluate_all(data.y_test, scores, target_fpr=cfg.evaluation.target_fpr)
    metrics.update({"model": "diffusion (ours)", "supervised": False, "train_time_s": train_time})
    results.append(metrics)
    print(f"  diffusion: {metrics}")

    # --- Random Forest (supervised) ---
    print("\n--- Training Random Forest baseline ---")
    t0 = time.time()
    rf = RandomForestBaseline(
        n_estimators=cfg.baselines.random_forest.n_estimators,
        max_depth=cfg.baselines.random_forest.max_depth,
        min_samples_leaf=cfg.baselines.random_forest.get("min_samples_leaf", 1),
        class_weight=cfg.baselines.random_forest.get("class_weight", None),
        seed=cfg.seed,
    ).fit(data.X_train_full, data.y_train_full)
    train_time = time.time() - t0
    scores = rf.anomaly_score(data.X_test)
    metrics = evaluate_all(data.y_test, scores, target_fpr=cfg.evaluation.target_fpr)
    metrics.update({"model": "random_forest", "supervised": True, "train_time_s": train_time})
    results.append(metrics)
    print(f"  random_forest: {metrics}")

    # --- Feedforward NN (supervised) ---
    print("\n--- Training feedforward NN baseline ---")
    t0 = time.time()
    nn_model = FeedforwardNNBaseline(
        in_dim=data.X_train_full.shape[1],
        hidden_dim=cfg.baselines.feedforward_nn.hidden_dim,
        n_hidden_layers=cfg.baselines.feedforward_nn.n_hidden_layers,
        lr=cfg.baselines.feedforward_nn.lr,
        batch_size=cfg.baselines.feedforward_nn.batch_size,
        epochs=cfg.baselines.feedforward_nn.epochs,
        seed=cfg.seed,
    ).fit(data.X_train_full, data.y_train_full)
    train_time = time.time() - t0
    scores = nn_model.anomaly_score(data.X_test)
    metrics = evaluate_all(data.y_test, scores, target_fpr=cfg.evaluation.target_fpr)
    metrics.update({"model": "feedforward_nn", "supervised": True, "train_time_s": train_time})
    results.append(metrics)
    print(f"  feedforward_nn: {metrics}")

    # --- Isolation Forest (unsupervised) ---
    print("\n--- Training Isolation Forest baseline ---")
    t0 = time.time()
    iso = IsolationForestBaseline(
        n_estimators=cfg.baselines.isolation_forest.n_estimators,
        contamination=cfg.baselines.isolation_forest.contamination,
        seed=cfg.seed,
    ).fit(data.X_train_normal)
    train_time = time.time() - t0
    scores = iso.anomaly_score(data.X_test)
    metrics = evaluate_all(data.y_test, scores, target_fpr=cfg.evaluation.target_fpr)
    metrics.update({"model": "isolation_forest", "supervised": False, "train_time_s": train_time})
    results.append(metrics)
    print(f"  isolation_forest: {metrics}")

    # --- Autoencoder (unsupervised) ---
    print("\n--- Training Autoencoder baseline ---")
    t0 = time.time()
    ae = AutoencoderBaseline(
        in_dim=data.X_train_normal.shape[1],
        hidden_dim=cfg.baselines.autoencoder.hidden_dim,
        latent_dim=cfg.baselines.autoencoder.latent_dim,
        lr=cfg.baselines.autoencoder.lr,
        batch_size=cfg.baselines.autoencoder.batch_size,
        epochs=cfg.baselines.autoencoder.epochs,
        seed=cfg.seed,
    ).fit(data.X_train_normal)
    train_time = time.time() - t0
    scores = ae.anomaly_score(data.X_test)
    metrics = evaluate_all(data.y_test, scores, target_fpr=cfg.evaluation.target_fpr)
    metrics.update({"model": "autoencoder", "supervised": False, "train_time_s": train_time})
    results.append(metrics)
    print(f"  autoencoder: {metrics}")

    df = pd.DataFrame(results)
    cols = ["model", "supervised", "auroc", "average_precision"] + [
        c for c in df.columns if c.startswith("f1_at_")
    ] + ["train_time_s"]
    df = df[cols]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="creditcard",
        choices=["creditcard", "bank_account_fraud", "paysim"],
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--seed", type=int, default=None, help="Override configs/default.yaml's seed for a single run.")
    parser.add_argument("--t-star", type=int, default=None, help="Override configs/default.yaml's diffusion.t_star.")
    args = parser.parse_args()

    df = run(args.dataset, args.config, seed=args.seed, t_star=args.t_star)

    print("\n=== Results ===")
    print(df.to_string(index=False))

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / f"results_{args.dataset}.csv"
    df.to_csv(out_csv, index=False)

    out_json = results_dir / f"results_{args.dataset}.json"
    with open(out_json, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    print(f"\nSaved results to {out_csv} and {out_json}")


if __name__ == "__main__":
    main()
