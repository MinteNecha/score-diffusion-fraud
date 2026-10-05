"""Check how much choosing t* on the test set inflates the diffusion results.

The test split is cut into two stratified halves. t* is chosen on one half by
Average Precision and the model is scored on the other half, then the roles
are swapped and the two scores are averaged. This is repeated per seed.

Usage:
    python scripts/validate_tstar.py --dataset creditcard
    python scripts/validate_tstar.py --dataset bank_account_fraud --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.data.loaders import get_dataset  # noqa: E402
from src.training.train_diffusion import train_diffusion_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402

T_GRID = [1, 2, 3, 5, 8, 12, 20, 30, 40, 50, 60, 70, 80, 90, 94, 99]

def one_seed(dataset: str, config: str, seed: int, t_grid: list[int], max_train_rows: int | None = None) -> dict:
    cfg = load_config(config, overrides={"dataset": {"name": dataset}, "seed": seed})
    data = get_dataset(
        dataset,
        raw_dir=PROJECT_ROOT / cfg.dataset.raw_dir,
        test_size=cfg.dataset.test_size,
        val_size=cfg.dataset.val_size,
        seed=seed,
    )
    X_train = data.X_train_normal
    if max_train_rows and len(X_train) > max_train_rows:
        rng = np.random.default_rng(seed)
        X_train = X_train[rng.choice(len(X_train), max_train_rows, replace=False)]
    d = cfg.diffusion
    model = train_diffusion_model(
        X_train, X_val_normal=data.X_val_normal,
        hidden_dim=d.hidden_dim, n_hidden_layers=d.n_hidden_layers, time_embed_dim=d.time_embed_dim,
        timesteps=d.timesteps, beta_start=d.beta_start, beta_end=d.beta_end, lr=d.lr,
        batch_size=d.batch_size, epochs=d.epochs, weight_decay=d.weight_decay, seed=seed,
    )
    y = data.y_test
    x = torch.tensor(data.X_test, dtype=torch.float32)
    scores = {t: model.anomaly_score(x, t_star=t, n_repeats=5).numpy() for t in t_grid}

    idx = np.arange(len(y))
    half_a, half_b = train_test_split(idx, test_size=0.5, stratify=y, random_state=seed)

    def ap(t, ids):
        return average_precision_score(y[ids], scores[t][ids])

    folds = []
    for pick, evaluate in ((half_a, half_b), (half_b, half_a)):
        t_pick = max(t_grid, key=lambda t: ap(t, pick))
        t_best_eval = max(t_grid, key=lambda t: ap(t, evaluate))
        folds.append((t_pick, ap(t_pick, evaluate), ap(t_best_eval, evaluate)))

    t_full = max(t_grid, key=lambda t: average_precision_score(y, scores[t]))
    return {
        "seed": seed,
        "t_full_test": t_full,
        "ap_full_test_best": average_precision_score(y, scores[t_full]),
        "t_picked_fold1": folds[0][0], "t_picked_fold2": folds[1][0],
        "ap_honest": float(np.mean([f[1] for f in folds])),
        "ap_oracle_halves": float(np.mean([f[2] for f in folds])),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["creditcard", "bank_account_fraud", "paysim"])
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--t-grid", type=int, nargs="+", default=T_GRID)
    p.add_argument("--max-train-rows", type=int, default=None, help="Cap on normal training rows (for large datasets).")
    args = p.parse_args()

    rows = [one_seed(args.dataset, args.config, s, args.t_grid, args.max_train_rows) for s in args.seeds]
    df = pd.DataFrame(rows)
    print("\n=== Per seed ===")
    print(df.to_string(index=False))
    print("\n=== Summary (mean +/- std over seeds) ===")
    for col, label in (
        ("ap_full_test_best", "AP with t* picked on the full test set (current method)"),
        ("ap_oracle_halves", "AP with t* picked on the same half it is scored on"),
        ("ap_honest", "AP with t* picked on the other half (honest)"),
    ):
        print(f"{label}: {df[col].mean():.4f} +/- {df[col].std(ddof=1):.4f}")

    out = PROJECT_ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / f"validate_tstar_{args.dataset}.csv", index=False)
    print(f"\nSaved to {out / f'validate_tstar_{args.dataset}.csv'}")


if __name__ == "__main__":
    main()