#!/usr/bin/env python3
"""Command line interface for the diffusion fraud detector.

    python cli.py train --dataset creditcard --t-star 94
    python cli.py score --dataset creditcard

`train` fits the diffusion model on normal transactions only and saves a
checkpoint. `score` loads that checkpoint, scores the held-out test split,
and writes per-transaction anomaly scores plus metrics. Every run appends a
telemetry record (timings, sizes, losses, throughput) to results/telemetry.jsonl.
The data split is rebuilt from the seed stored in the checkpoint, so the test
set is the same one the model was never trained on.
"""
from __future__ import annotations

import argparse 
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.loaders import get_dataset
from src.evaluation.metrics import evaluate_all
from src.models.diffusion import GaussianDiffusion, MLPDenoiser
from src.training.train_diffusion import train_diffusion_model
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parent
DATASETS = ["creditcard", "bank_account_fraud", "paysim"]


def log_telemetry(cfg, record: dict) -> None:
    out = ROOT / cfg.output.results_dir
    out.mkdir(parents=True, exist_ok=True)
    record["time_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(out / "telemetry.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    print("telemetry:", json.dumps(record))


def build_config(args):
    overrides = {"dataset": {"name": args.dataset}}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if getattr(args, "t_star", None) is not None:
        overrides["diffusion"] = {"t_star": args.t_star}
    return load_config(args.config, overrides=overrides)


def load_data(cfg):
    return get_dataset(
        cfg.dataset.name,
        raw_dir=ROOT / cfg.dataset.raw_dir,
        test_size=cfg.dataset.test_size,
        val_size=cfg.dataset.val_size,
        seed=cfg.seed,
    )


def checkpoint_path(cfg) -> Path:
    return ROOT / cfg.output.model_dir / f"diffusion_{cfg.dataset.name}.pt"


def train(args) -> None:
    cfg = build_config(args)
    d = cfg.diffusion
    t0 = time.time()
    data = load_data(cfg)
    load_s = time.time() - t0
    print(f"train_normal={data.X_train_normal.shape} val_normal={data.X_val_normal.shape} test={data.X_test.shape}")

    t0 = time.time()
    model = train_diffusion_model(
        data.X_train_normal,
        X_val_normal=data.X_val_normal,
        hidden_dim=d.hidden_dim,
        n_hidden_layers=d.n_hidden_layers,
        time_embed_dim=d.time_embed_dim,
        timesteps=d.timesteps,
        beta_start=d.beta_start,
        beta_end=d.beta_end,
        lr=d.lr,
        batch_size=d.batch_size,
        epochs=d.epochs,
        weight_decay=d.weight_decay,
        seed=cfg.seed,
    )
    train_s = time.time() - t0

    model.model.eval()
    with torch.no_grad():
        val_loss = model.training_loss(torch.tensor(data.X_val_normal, dtype=torch.float32)).item()

    path = checkpoint_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": dict(cfg), "n_features": data.X_train_normal.shape[1]}, path)
    print(f"saved checkpoint to {path}")

    log_telemetry(cfg, {
        "command": "train", "dataset": cfg.dataset.name, "seed": cfg.seed, "t_star": d.t_star,
        "n_train_normal": len(data.X_train_normal), "n_features": data.X_train_normal.shape[1],
        "n_parameters": sum(p.numel() for p in model.model.parameters()),
        "load_seconds": round(load_s, 2), "train_seconds": round(train_s, 2),
        "final_val_loss": round(val_loss, 5), "checkpoint_mb": round(path.stat().st_size / 1e6, 3),
    })


def score(args) -> None:
    cfg = build_config(args)
    path = checkpoint_path(cfg)
    if not path.exists():
        raise SystemExit(f"No checkpoint at {path}. Run: python cli.py train --dataset {cfg.dataset.name}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    saved = load_config(args.config, overrides=ckpt["config"])
    d = saved.diffusion
    if args.t_star is not None:
        d.t_star = args.t_star

    data = load_data(saved)
    net = MLPDenoiser(data_dim=ckpt["n_features"], hidden_dim=d.hidden_dim,
                      n_hidden_layers=d.n_hidden_layers, time_embed_dim=d.time_embed_dim)
    model = GaussianDiffusion(net, timesteps=d.timesteps, beta_start=d.beta_start, beta_end=d.beta_end)
    model.load_state_dict(ckpt["state_dict"])

    x = torch.tensor(data.X_test, dtype=torch.float32)
    t0 = time.time()
    scores = model.anomaly_score(x, t_star=d.t_star, n_repeats=args.n_repeats).numpy()
    score_s = time.time() - t0

    metrics = evaluate_all(data.y_test, scores, target_fpr=saved.evaluation.target_fpr)
    out = ROOT / saved.output.results_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"anomaly_score": scores, "is_fraud": data.y_test}).to_csv(out / f"scores_{saved.dataset.name}.csv", index=False)
    with open(out / f"metrics_{saved.dataset.name}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    log_telemetry(saved, {
        "command": "score", "dataset": saved.dataset.name, "t_star": d.t_star, "n_repeats": args.n_repeats,
        "n_test": len(scores), "score_seconds": round(score_s, 3),
        "microseconds_per_transaction": round(score_s / len(scores) * 1e6, 2), **{k: round(float(v), 4) for k, v in metrics.items()},
    })


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn in (("train", train), ("score", score)):
        s = sub.add_parser(name)
        s.add_argument("--dataset", required=True, choices=DATASETS)
        s.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
        s.add_argument("--seed", type=int, default=None)
        s.add_argument("--t-star", type=int, default=None, help="Noise level (creditcard 94, paysim 5, bank_account_fraud 1).")
        if name == "score":
            s.add_argument("--n-repeats", type=int, default=5, help="Noise draws averaged per transaction.")
        s.set_defaults(func=fn)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()