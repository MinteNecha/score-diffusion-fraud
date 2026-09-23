#!/usr/bin/env python3
"""Download the three project datasets from Kaggle into data/raw/.

Requires the Kaggle API to be set up: pip install kaggle, and a valid
~/.kaggle/kaggle.json API token (see https://www.kaggle.com/docs/api).

Usage:
    python scripts/download_data.py                  # download all three
    python scripts/download_data.py --dataset creditcard
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# (kaggle dataset slug, expected primary csv filename after unzip)
DATASETS = {
    "creditcard": ("mlg-ulb/creditcardfraud", "creditcard.csv"),
    "bank_account_fraud": ("sgpjesus/bank-account-fraud-dataset-neurips-2022", "Base.csv"),
    "paysim": ("ealaxi/paysim1", "PS_20174392719_1491204439457_log.csv"),
}


def _check_kaggle_cli() -> None:
    if shutil.which("kaggle") is None:
        sys.exit(
            "The 'kaggle' CLI was not found. Install it with `pip install kaggle` "
            "and place your API token at ~/.kaggle/kaggle.json, then re-run this script."
        )


def download_one(name: str) -> None:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASETS)}.")
    slug, expected_csv = DATASETS[name]

    dest_dir = RAW_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    if (dest_dir / expected_csv).exists():
        print(f"[{name}] {expected_csv} already present in {dest_dir}, skipping download.")
        return

    print(f"[{name}] downloading {slug} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest_dir)],
        check=True,
    )

    zip_path = dest_dir / f"{slug.split('/')[-1]}.zip"
    if not zip_path.exists():
        # kaggle sometimes names the zip differently; find whatever zip just landed
        zips = sorted(dest_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not zips:
            sys.exit(f"[{name}] download finished but no zip file was found in {dest_dir}.")
        zip_path = zips[0]

    print(f"[{name}] extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()

    if not (dest_dir / expected_csv).exists():
        print(
            f"[{name}] WARNING: expected '{expected_csv}' after extraction but it wasn't found. "
            f"Check {dest_dir} and rename the CSV, or update src/data/loaders.py to match."
        )
    else:
        print(f"[{name}] done -> {dest_dir / expected_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS) + ["all"],
        default="all",
        help="Which dataset to download (default: all three).",
    )
    args = parser.parse_args()

    _check_kaggle_cli()

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        download_one(name)


if __name__ == "__main__":
    main()
