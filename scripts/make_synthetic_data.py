"""Generate small synthetic CSVs matching each dataset's real schema, for
smoke-testing the pipeline without downloading the real (large) Kaggle
data. NOT used for actual experiments/results - synthetic only.

Usage:
    python scripts/make_synthetic_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

RAW_DIR = PROJECT_ROOT / "data" / "raw"


def make_creditcard(n: int = 2000, fraud_rate: float = 0.02, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n * fraud_rate))
    n_normal = n - n_fraud

    cols = [f"V{i}" for i in range(1, 29)]
    normal = rng.normal(0, 1, size=(n_normal, 28))
    fraud = rng.normal(0, 1, size=(n_fraud, 28)) + rng.normal(3, 1, size=28)  # shifted distribution

    df = pd.DataFrame(np.vstack([normal, fraud]), columns=cols)
    df["Time"] = rng.integers(0, 172800, size=n)
    df["Amount"] = np.abs(rng.normal(50, 40, size=n))
    df["Class"] = np.array([0] * n_normal + [1] * n_fraud)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def make_bank_account_fraud(n: int = 2000, fraud_rate: float = 0.02, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n * fraud_rate))
    n_normal = n - n_fraud
    n_total = n_normal + n_fraud

    df = pd.DataFrame(
        {
            "income": rng.uniform(0, 1, n_total),
            "name_email_similarity": rng.uniform(0, 1, n_total),
            "prev_address_months_count": rng.integers(-1, 380, n_total),
            "current_address_months_count": rng.integers(-1, 430, n_total),
            "customer_age": rng.integers(18, 90, n_total),
            "days_since_request": rng.uniform(0, 79, n_total),
            "intended_balcon_amount": rng.normal(0, 50, n_total),
            "payment_type": rng.choice(["AA", "AB", "AC", "AD"], n_total),
            "zip_count_4w": rng.integers(1, 6000, n_total),
            "velocity_6h": rng.normal(5000, 3000, n_total),
            "velocity_24h": rng.normal(5000, 3000, n_total),
            "velocity_4w": rng.normal(5000, 3000, n_total),
            "bank_branch_count_8w": rng.integers(0, 2400, n_total),
            "date_of_birth_distinct_emails_4w": rng.integers(0, 39, n_total),
            "employment_status": rng.choice(["CA", "CB", "CC", "CD"], n_total),
            "credit_risk_score": rng.integers(-170, 390, n_total),
            "email_is_free": rng.integers(0, 2, n_total),
            "housing_status": rng.choice(["BA", "BB", "BC"], n_total),
            "phone_home_valid": rng.integers(0, 2, n_total),
            "phone_mobile_valid": rng.integers(0, 2, n_total),
            "bank_months_count": rng.integers(-1, 32, n_total),
            "has_other_cards": rng.integers(0, 2, n_total),
            "proposed_credit_limit": rng.choice([200, 500, 1000, 1500, 2000], n_total),
            "foreign_request": rng.integers(0, 2, n_total),
            "source": rng.choice(["INTERNET", "TELEAPP"], n_total),
            "session_length_in_minutes": rng.uniform(-1, 85, n_total),
            "device_os": rng.choice(["windows", "macintosh", "linux", "other"], n_total),
            "keep_alive_session": rng.integers(0, 2, n_total),
            "device_distinct_emails_8w": rng.integers(-1, 3, n_total),
            "device_fraud_count": np.zeros(n_total, dtype=int),
            "month": rng.integers(0, 8, n_total),
        }
    )
    df["fraud_bool"] = np.array([0] * n_normal + [1] * n_fraud)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def make_paysim(n: int = 2000, fraud_rate: float = 0.02, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n * fraud_rate))
    n_normal = n - n_fraud
    n_total = n_normal + n_fraud

    amount = np.abs(rng.normal(5000, 4000, n_total))
    old_orig = np.abs(rng.normal(10000, 8000, n_total))
    new_orig = np.clip(old_orig - amount, 0, None)
    old_dest = np.abs(rng.normal(5000, 5000, n_total))
    new_dest = old_dest + amount

    df = pd.DataFrame(
        {
            "step": rng.integers(1, 744, n_total),
            "type": rng.choice(["CASH_OUT", "PAYMENT", "TRANSFER", "CASH_IN", "DEBIT"], n_total),
            "amount": amount,
            "nameOrig": [f"C{i}" for i in range(n_total)],
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": new_orig,
            "nameDest": [f"M{i}" for i in range(n_total)],
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "isFlaggedFraud": np.zeros(n_total, dtype=int),
        }
    )
    df["isFraud"] = np.array([0] * n_normal + [1] * n_fraud)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    make_creditcard().to_csv(RAW_DIR / "creditcard.csv", index=False)
    print(f"wrote {RAW_DIR / 'creditcard.csv'}")

    make_bank_account_fraud().to_csv(RAW_DIR / "Base.csv", index=False)
    print(f"wrote {RAW_DIR / 'Base.csv'}")

    make_paysim().to_csv(RAW_DIR / "PS_20174392719_1491204439457_log.csv", index=False)
    print(f"wrote {RAW_DIR / 'PS_20174392719_1491204439457_log.csv'}")


if __name__ == "__main__":
    main()
