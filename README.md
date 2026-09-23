# Score-Based Diffusion Models for Unsupervised Financial Transaction Anomaly Detection

**Author:** Mintesinot Zemade Necha — University of Johannesburg
**Module:** IT18X57 Advanced Artificial Intelligence

## What this project does

A diffusion model is trained **only on normal transactions** (no fraud labels
used at training time). At test time, a transaction is noised to a fixed
level `t*`, the model tries to denoise it back toward the normal
distribution it learned, and the reconstruction error becomes the anomaly
score. A normal transaction is easy to restore; a fraudulent one isn't,
since it doesn't match what the model learned — so it gets a high score.

This is compared against four baselines: two supervised (Random Forest,
feedforward neural network — both need fraud labels during training) and
two unsupervised (Isolation Forest, standard Autoencoder — same
label-free setting as the diffusion model). The comparison answers the
project's research question: **what does it cost, in detection
performance, to remove the need for fraud labels entirely?**

The approach is evaluated across three datasets of increasing difficulty:

| Dataset | Role | Size | Features |
|---|---|---|---|
| `creditcard` (ULB Credit Card Fraud) | Primary | 284,807 rows | 30 (PCA'd V1–V28, Time, Amount) |
| `bank_account_fraud` (NeurIPS 2022 suite, Base variant) | Secondary — generalisation check | 1,000,000 rows | 30 (mixed numeric/categorical) |
| `paysim` (Synthetic Financial Datasets) | Tertiary — cross-domain stress test | 6,362,620 rows | 11 (mixed, several categorical) |

An additional experiment (`scripts/sweep_tstar.py`) investigates how the
choice of noise level `t*` affects detection performance — something the
anchor paper (Livernoche et al., *On Diffusion Modeling for Anomaly
Detection*, ICLR 2024 Spotlight) didn't cover for financial data.

## Project structure

```
score-diffusion-fraud/
  configs/
    default.yaml           # all hyperparameters in one place
  data/
    raw/                   # put downloaded Kaggle CSVs here
    processed/              # (reserved for cached/preprocessed data, currently unused)
  results/                  # metrics CSV/JSON + checkpoints land here
  scripts/
    download_data.py        # pulls all three datasets from Kaggle
    run_experiment.py        # trains + evaluates diffusion model vs all baselines, one dataset
    sweep_tstar.py            # trains once, evaluates across many noise levels t*
  src/
    data/loaders.py          # per-dataset loading -> common DatasetSplits interface
    models/diffusion.py      # MLPDenoiser + GaussianDiffusion (DDPM forward/reverse process)
    models/baselines.py      # RandomForest / FeedforwardNN / IsolationForest / Autoencoder
    training/train_diffusion.py  # training loop for the diffusion model
    evaluation/metrics.py    # AUROC, Average Precision, F1 at 1% FPR
    utils/config.py          # tiny YAML config loader (attribute-style access)
    utils/seed.py             # reproducibility helper
  notebooks/                 # (empty — for exploratory analysis / plots)
```

## Setup

```bash
pip install -r requirements.txt
```

To download the datasets, you need a Kaggle API token at
`~/.kaggle/kaggle.json` (see the [Kaggle API docs](https://www.kaggle.com/docs/api)),
then:

```bash
python scripts/download_data.py               # all three datasets
python scripts/download_data.py --dataset creditcard   # just one
```

This populates `data/raw/` with `creditcard.csv`, `Base.csv`, and
`PS_20174392719_1491204439457_log.csv`.

## Running an experiment

```bash
# Full comparison: diffusion model vs. all four baselines, on one dataset
python scripts/run_experiment.py --dataset creditcard

# The t* noise-level investigation (trains the diffusion model once,
# evaluates it at many different noise levels)
python scripts/sweep_tstar.py --dataset creditcard --t-values 5 10 20 30 50 70 90
```

Results are printed to the console and saved under `results/` as both
CSV and JSON.

To run on the other two datasets, swap `--dataset creditcard` for
`--dataset bank_account_fraud` or `--dataset paysim`. All hyperparameters
live in `configs/default.yaml` — edit them there rather than hardcoding
changes in the scripts.

## Smoke-testing without real data

Before running on the real (large) Kaggle datasets, you can sanity-check
that every module imports and runs end-to-end using small random arrays.
See `scripts/` for the real entry points — a synthetic-data smoke test
can be run directly against `src/` without downloading anything (useful
for quickly checking the pipeline still works after a code change).

## Evaluation metrics

- **AUROC** — overall ability to rank fraud above normal transactions.
- **Average Precision** — area under the precision-recall curve; more
  informative than AUROC here because fraud is a tiny fraction of the data
  (~0.17% in the primary dataset), so AUROC alone can look artificially high.
- **F1 at 1% FPR** — F1 score at the operating point where only 1% of
  normal transactions get (falsely) flagged, simulating a realistic
  constraint on how many transactions a fraud team can actually review.

## References

1. Dal Pozzolo, A. et al. — ULB Credit Card Fraud Dataset. Kaggle (2015).
2. Ho, J., Jain, A., Abbeel, P. — Denoising Diffusion Probabilistic Models. NeurIPS 33 (2020).
3. Livernoche, V., Jain, V., Hezaveh, Y., Ravanbakhsh, S. — On Diffusion Modeling for Anomaly Detection. ICLR Spotlight (2024).
4. Watson, D. et al. — FraudDiffuse: Diffusion-aided Synthetic Fraud Augmentation for Improved Fraud Detection. ACM ICAIF (2024).
