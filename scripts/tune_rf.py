#!/usr/bin/env python3
"""Tune the Random Forest baseline per dataset via cross-validated grid search.

Why this exists: the diffusion model gets a real tuning pass (t* swept
exhaustively per dataset in sweep_tstar.py); until now Random Forest ran on
guessed defaults (n_estimators=200, max_depth=None), which is an unfair
comparison - RF is known to be fairly hyperparameter-sensitive, especially
on imbalanced data. This script closes that gap for RF specifically, since
it's the strongest baseline and the main point of comparison in the "cost
of label-free detection" analysis.

Method: GridSearchCV with stratified k-fold CV, scored on average_precision
(the project's primary metric, consistent with how every other result is
judged). Only X_train_full/y_train_full is touched - the held-out X_test
set is never used for tuning, so nothing leaks into the final reported
numbers.

Usage:
    python scripts/tune_rf.py --dataset creditcard
    python scripts/tune_rf.py --dataset bank_account_fraud
    python scripts/tune_rf.py --dataset paysim

Writes results/rf_best_params_<dataset>.json, which run_experiment.py and
run_multiseed.py pick up automatically if present.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402

from src.data.loaders import get_dataset  # noqa: E402
from src.utils.config import load_config  # noqa: E402

# Kept deliberately small - this already means n_splits * len(grid) forest
# fits (3 * 16 = 48 below), and two of the three datasets have 300k-700k
# rows in X_train_full.
PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [None, 20],
    "min_samples_leaf": [1, 5],
    "class_weight": [None, "balanced"],
}


def tune(dataset_name: str, config_path: str, seed: int = 42, n_splits: int = 3) -> dict:
    cfg = load_config(config_path, overrides={"dataset": {"name": dataset_name}})
    data = get_dataset(
        dataset_name,
        raw_dir=PROJECT_ROOT / cfg.dataset.raw_dir,
        test_size=cfg.dataset.test_size,
        val_size=cfg.dataset.val_size,
        seed=seed,
    )
    print(
        f"[{dataset_name}] tuning on X_train_full={data.X_train_full.shape} "
        f"(X_test is never touched here)"
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    base = RandomForestClassifier(random_state=seed, n_jobs=1)
    search = GridSearchCV(
        base, PARAM_GRID, scoring="average_precision", cv=cv, n_jobs=-1, refit=False
    )

    t0 = time.time()
    search.fit(data.X_train_full, data.y_train_full)
    elapsed = time.time() - t0

    best_params = dict(search.best_params_)
    print(f"[{dataset_name}] grid search took {elapsed:.1f}s")
    print(f"[{dataset_name}] best CV average_precision = {search.best_score_:.4f}")
    print(f"[{dataset_name}] best params = {best_params}")
    return best_params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, choices=["creditcard", "bank_account_fraud", "paysim"]
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=3)
    args = parser.parse_args()

    best_params = tune(args.dataset, args.config, args.seed, args.n_splits)

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rf_best_params_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved best params to {out_path}")


if __name__ == "__main__":
    main()