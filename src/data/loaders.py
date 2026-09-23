"""Unified dataset loading for the three fraud-detection datasets used in
this project:

  1. creditcard          - ULB Credit Card Fraud (primary)
                            https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  2. bank_account_fraud  - Bank Account Fraud Dataset Suite, NeurIPS 2022, Base variant
                            https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
  3. paysim              - Synthetic Financial Datasets For Fraud Detection
                            https://www.kaggle.com/datasets/ealaxi/paysim1

Every loader returns a `DatasetSplits` object with a *common* interface so
the rest of the pipeline (diffusion model + baselines + evaluation) never
needs to know which raw dataset it is looking at:

  - X_train_normal : features for unsupervised training, NORMAL transactions only
  - X_val_normal   : held-out normal transactions, for early stopping / t* selection
  - X_test         : mixed normal + fraud test set (features)
  - y_test         : labels for X_test (0 = normal, 1 = fraud)
  - X_train_full / y_train_full : full labelled training split, for the
                      *supervised* baselines (Random Forest, feedforward NN)
  - feature_names  : list of column names, in the order used by the matrices

All feature matrices are float32 numpy arrays, scaled with a StandardScaler
fit on the training-normal data only (to avoid leaking test-set statistics).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclasses.dataclass
class DatasetSplits:
    X_train_normal: np.ndarray   # (n, d) float32 - normal only, for diffusion training
    X_val_normal: np.ndarray     # (n_val, d) float32 - normal only, held out
    X_test: np.ndarray           # (n_test, d) float32 - normal + fraud
    y_test: np.ndarray           # (n_test,) int, 0=normal 1=fraud
    X_train_full: np.ndarray     # (n_full, d) float32 - normal + fraud, for supervised baselines
    y_train_full: np.ndarray     # (n_full,) int, labels for X_train_full
    feature_names: List[str]
    dataset_name: str
    fraud_rate: float            # fraction of fraud in the full dataset (before splitting)


def _make_splits(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    dataset_name: str,
    test_size: float,
    val_size: float,
    seed: int,
) -> DatasetSplits:
    """Shared split + scale logic used by every per-dataset loader.

    Strategy:
      1. Stratified train/test split so the test set keeps a realistic
         fraud rate and contains ALL the signal needed to evaluate.
      2. From the training portion, take only the normal rows and split
         those again into train/val (the diffusion model never sees fraud
         or labels during training - that's the whole point).
      3. A StandardScaler is fit on X_train_normal only, then applied to
         everything else, so no test statistics leak into training.
    """
    fraud_rate = float(y.mean())

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    normal_mask = y_train_full == 0
    X_train_normal_all = X_train_full[normal_mask]

    X_train_normal, X_val_normal = train_test_split(
        X_train_normal_all, test_size=val_size, random_state=seed
    )

    scaler = StandardScaler()
    scaler.fit(X_train_normal)

    X_train_normal = scaler.transform(X_train_normal).astype(np.float32)
    X_val_normal = scaler.transform(X_val_normal).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    X_train_full = scaler.transform(X_train_full).astype(np.float32)

    return DatasetSplits(
        X_train_normal=X_train_normal,
        X_val_normal=X_val_normal,
        X_test=X_test,
        y_test=y_test.astype(np.int64),
        X_train_full=X_train_full,
        y_train_full=y_train_full.astype(np.int64),
        feature_names=feature_names,
        dataset_name=dataset_name,
        fraud_rate=fraud_rate,
    )


def load_creditcard(
    raw_dir: str | Path,
    test_size: float = 0.3,
    val_size: float = 0.1,
    seed: int = 42,
) -> DatasetSplits:
    """ULB Credit Card Fraud dataset (primary).

    Expects `creditcard.csv` in `raw_dir` with columns:
    Time, V1..V28, Amount, Class (1 = fraud).
    """
    path = Path(raw_dir) / "creditcard.csv"
    df = pd.read_csv(path)

    y = df["Class"].to_numpy()
    df = df.drop(columns=["Class"])

    # Time and Amount are on very different scales to the PCA'd V1-V28
    # columns; StandardScaler (applied later in _make_splits) handles this,
    # but Amount benefits from a log1p first since it's heavily skewed.
    df["Amount"] = np.log1p(df["Amount"])

    feature_names = list(df.columns)
    X = df.to_numpy(dtype=np.float64)

    return _make_splits(X, y, feature_names, "creditcard", test_size, val_size, seed)


def load_bank_account_fraud(
    raw_dir: str | Path,
    test_size: float = 0.3,
    val_size: float = 0.1,
    seed: int = 42,
    variant: str = "Base",
) -> DatasetSplits:
    """Bank Account Fraud Dataset Suite (NeurIPS 2022), Base variant (secondary).

    Expects a CSV such as `Base.csv` in `raw_dir` (the NeurIPS 2022 suite
    ships one CSV per variant: Base, Variant I..V). Label column is
    `fraud_bool`. Several columns are categorical and get one-hot encoded;
    a handful of columns use -1 as a "missing" sentinel, which we leave as
    a normal numeric value (the model can learn it's an out-of-range code).
    """
    candidates = [Path(raw_dir) / f"{variant}.csv", Path(raw_dir) / "bank_account_fraud.csv"]
    path = next((p for p in candidates if p.exists()), candidates[0])
    df = pd.read_csv(path)

    y = df["fraud_bool"].to_numpy()
    df = df.drop(columns=["fraud_bool"])

    # Drop columns that are purely bookkeeping / not useful signal for an
    # unsupervised model (month is a temporal split key used by the dataset
    # authors for their own train/test protocol, not a transaction feature).
    for col in ["month"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Catch both classic numpy "object" string columns and pandas' newer
    # dedicated string dtype (the default for text columns since pandas 3.0).
    categorical_cols = list(df.select_dtypes(include=["object", "string"]).columns)
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)

    # get_dummies can produce bool columns; cast everything to float
    feature_names = list(df.columns)
    X = df.to_numpy(dtype=np.float64)

    return _make_splits(X, y, feature_names, "bank_account_fraud", test_size, val_size, seed)


def load_paysim(
    raw_dir: str | Path,
    test_size: float = 0.3,
    val_size: float = 0.1,
    seed: int = 42,
    max_rows: Optional[int] = None,
) -> DatasetSplits:
    """PaySim synthetic mobile-money dataset (tertiary / cross-domain stress test).

    Expects `PS_20174392719_1491204439457_log.csv` (or any CSV placed as
    `paysim.csv`) in `raw_dir`. Label column is `isFraud`. `nameOrig` and
    `nameDest` are transaction-unique identifiers (not useful, and would
    just balloon the one-hot encoding) and are dropped; `type` is a small
    categorical column and is one-hot encoded. `isFlaggedFraud` is a
    rule-based flag set by the simulator itself and is dropped to avoid
    leaking a near-perfect proxy for the label.

    `max_rows` optionally subsamples the (very large, ~6.4M row) file for
    faster experimentation - pass None to use the full dataset.
    """
    candidates = [
        Path(raw_dir) / "paysim.csv",
        Path(raw_dir) / "PS_20174392719_1491204439457_log.csv",
    ]
    path = next((p for p in candidates if p.exists()), candidates[0])
    df = pd.read_csv(path, nrows=max_rows)

    y = df["isFraud"].to_numpy()
    df = df.drop(columns=["isFraud"])

    for col in ["nameOrig", "nameDest", "isFlaggedFraud"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = pd.get_dummies(df, columns=["type"], dummy_na=False)

    feature_names = list(df.columns)
    X = df.to_numpy(dtype=np.float64)

    return _make_splits(X, y, feature_names, "paysim", test_size, val_size, seed)


_LOADERS = {
    "creditcard": load_creditcard,
    "bank_account_fraud": load_bank_account_fraud,
    "paysim": load_paysim,
}


def get_dataset(name: str, raw_dir: str | Path, **kwargs) -> DatasetSplits:
    """Dispatch to the right loader by dataset name.

    `name` must be one of "creditcard", "bank_account_fraud", "paysim".
    Extra kwargs (test_size, val_size, seed, ...) are forwarded to the
    specific loader.
    """
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(_LOADERS)}.")
    return _LOADERS[name](raw_dir, **kwargs)
