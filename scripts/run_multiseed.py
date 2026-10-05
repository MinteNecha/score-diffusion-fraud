"""Run the full model comparison across multiple random seeds and report
mean +/- std for every metric, instead of trusting a single noisy run.

Usage:
    python scripts/run_multiseed.py --dataset creditcard --seeds 42 43 44 45 46
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd  # noqa: E402

from run_experiment import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="creditcard",
        choices=["creditcard", "bank_account_fraud", "paysim"],
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--t-star", type=int, default=None, help="Override configs/default.yaml's diffusion.t_star.")
    args = parser.parse_args()

    all_runs = []
    for seed in args.seeds:
        print(f"\n################ seed={seed} ################")
        df = run(args.dataset, args.config, seed=seed, t_star=args.t_star)
        df["seed"] = seed
        all_runs.append(df)

    combined = pd.concat(all_runs, ignore_index=True)

    metric_cols = [c for c in combined.columns if c not in ("model", "supervised", "seed")]
    summary = combined.groupby(["model", "supervised"])[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / f"multiseed_raw_{args.dataset}.csv"
    summary_path = results_dir / f"multiseed_summary_{args.dataset}.csv"
    combined.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"\n=== Mean +/- std across seeds {args.seeds} ({args.dataset}) ===")
    print(summary.to_string(index=False))
    print(f"\nSaved raw per-seed results to {raw_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()