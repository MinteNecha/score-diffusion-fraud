# Score-Based Diffusion Models for Unsupervised Financial Transaction Anomaly Detection

## What this project does

A diffusion model is trained only on normal transactions, so no fraud labels
are used during training. At test time a transaction is noised to a fixed
level `t*`, the model denoises it in one step, and the reconstruction error
becomes the anomaly score. A normal transaction is easy to restore. A
fraudulent one does not match what the model learned, so it gets a high score.

This is compared against four baselines: two supervised (Random Forest and a
feedforward neural network, both trained with fraud labels) and two
unsupervised (Isolation Forest and an autoencoder, trained on normal data
only like the diffusion model). The research question is how much detection
performance is lost by removing the need for fraud labels.

The method follows the anchor paper by Livernoche et al., *On Diffusion
Modeling for Anomaly Detection* (ICLR 2024), applied to financial data.

## Datasets

| Dataset | Rows | Features | Notes |
|---|---|---|---|
| `creditcard` (ULB Credit Card Fraud) | 284,807 | 30 | 1,081 duplicate rows are dropped |
| `bank_account_fraud` (NeurIPS 2022, Base variant) | 1,000,000 | FEATURES_AFTER after preprocessing | `-1` codes treated as missing (indicator column plus median fill) |
| `paysim` (Synthetic Financial Datasets) | 6,362,620 | 11 | `isFlaggedFraud` is dropped |

All cleaning lives in `src/data/loaders.py`. The notebooks only contain the
exploratory analysis that justified each cleaning step.

## Project structure

```
score-diffusion-fraud/
  cli.py                      # train and score commands (start here)
  configs/
    default.yaml              # all hyperparameters in one place
    smoke.yaml                # tiny settings for quick pipeline checks
  data/raw/                   # downloaded CSVs go here (not in the repo)
  notebooks/                  # EDA for each dataset
  results/                    # metrics, scores, tuned parameters, figures
  scripts/
    download_data.py          # pulls the datasets from Kaggle
    run_experiment.py         # one run: diffusion vs all four baselines
    run_multiseed.py          # same comparison over several seeds, mean and std
    sweep_tstar.py            # trains once, scores at many noise levels t*
    validate_tstar.py         # checks how much picking t* on the test set inflates AP
    tune_rf.py                # Random Forest grid search (training split only)
    make_figures.py           # PR curves, score histograms, scoring speed
    make_synthetic_data.py    # small fake CSVs for smoke testing
  src/
    data/loaders.py           # loading, cleaning, splitting, scaling
    models/diffusion.py       # MLPDenoiser and GaussianDiffusion (DDPM)
    models/baselines.py       # RF, feedforward NN, Isolation Forest, autoencoder
    training/train_diffusion.py
    evaluation/metrics.py     # AUROC, Average Precision, F1 at 1% FPR
    utils/                    # config loader, seeding
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux or macOS
pip install -r requirements.txt
```

Downloading the data needs a Kaggle API token at `~/.kaggle/kaggle.json`
(see the Kaggle API docs):

```bash
python scripts/download_data.py                        # all three datasets
python scripts/download_data.py --dataset creditcard   # one dataset
```

This fills `data/raw/` with `creditcard.csv`, `Base.csv` and
`PS_20174392719_1491204439457_log.csv`.

## Quick start (train, then score)

```bash
python cli.py train --dataset creditcard --t-star 94
python cli.py score --dataset creditcard
```

`train` fits the diffusion model on normal transactions only and saves a
checkpoint to `results/checkpoints/`. `score` reloads it and scores the
held-out test split. Outputs in `results/`: `scores_<dataset>.csv` (one anomaly
score per test transaction), `metrics_<dataset>.json` (AUROC, Average Precision,
F1 at 1% FPR) and `telemetry.jsonl` (timings, parameter count, validation loss,
microseconds per transaction). Hyperparameters are set in `configs/default.yaml`,
or per run with `--t-star`, `--seed` and `--config`.

## Running the experiments

The best noise level found for each dataset is `t*` = 94 for `creditcard`,
5 for `paysim` and 1 for `bank_account_fraud`.

```bash
# One run, all five models, one dataset
python scripts/run_experiment.py --dataset creditcard --t-star 94

# Five seeds (42 to 46), mean and standard deviation. This is what the paper reports.
python scripts/run_multiseed.py --dataset creditcard --t-star 94
python scripts/run_multiseed.py --dataset bank_account_fraud --t-star 1
python scripts/run_multiseed.py --dataset paysim --t-star 5

# How the choice of t* changes detection performance
python scripts/sweep_tstar.py --dataset creditcard

# Does picking t* on the test set inflate the results? (two-fold check)
python scripts/validate_tstar.py --dataset creditcard

# Figures and scoring speed (saved to results/figures/)
python scripts/make_figures.py --dataset creditcard --t-star 94

# Random Forest hyperparameter search (3-fold CV on the training split)
python scripts/tune_rf.py --dataset creditcard
```

Results print to the console and are saved under `results/` as CSV and JSON.
All hyperparameters are in `configs/default.yaml`.

Approximate run times on CPU: `creditcard` about 3 to 4 minutes per run,
`paysim` about 49 minutes per run, because it has 4.45 million training rows.

## Quick check without the real data

```bash
python scripts/make_synthetic_data.py
python scripts/run_experiment.py --dataset creditcard --config configs/smoke.yaml
```

The synthetic data is only for checking that the code runs. It is not used for
any reported result.

## Evaluation metrics

- **AUROC**: how well the model ranks fraud above normal transactions.
- **Average Precision (AP)**: area under the precision-recall curve. This is the
  main metric, because fraud is a tiny fraction of the data and AUROC alone
  can look too good.
- **F1 at 1% FPR**: F1 at the point where only 1% of normal transactions are
  flagged, which mimics a limit on how many alerts a fraud team can review.

## Main results

Mean over five seeds, Average Precision.

| Dataset | Diffusion | Autoencoder | Isolation Forest | Random Forest | Neural net |
|---|---|---|---|---|---|
| creditcard | 0.549 | 0.616 | 0.129 | 0.838 | 0.731 |
| paysim | 0.333 | 0.241 | 0.007 | 0.947 | 0.636 |
| bank_account_fraud | 0.012 | 0.015 | 0.014 | 0.137 | 0.102 |

The label-free diffusion model reaches roughly two thirds of Random Forest's
AP on `creditcard`, about a third on `paysim`, and very little on
`bank_account_fraud`. See the paper for the full tables and discussion.

## Known limitations

- `t*` was chosen by maximising AP on the test split. A two-fold check
  (`scripts/validate_tstar.py`: pick `t*` on one half of the test set, score the
  other half, then swap) found the effect negligible on all three datasets
  (PaySim used one seed and a 300,000-row training sample). A fully separate
  labelled validation split would still be cleaner.
- Isolation Forest, the autoencoder and the neural network use standard default
  settings. Only Random Forest was tuned, and it gave a small gain, so the
  untuned Random Forest is reported for consistency.
- Results vary slightly between runs because PyTorch is not fully
  deterministic on CPU, which is why five seeds are used.

## References

1. Dal Pozzolo, A. et al. Calibrating probability with undersampling for unbalanced classification. IEEE SSCI (2015).
2. Ho, J., Jain, A., Abbeel, P. Denoising Diffusion Probabilistic Models. NeurIPS 33 (2020).
3. Livernoche, V., Jain, V., Hezaveh, Y., Ravanbakhsh, S. On Diffusion Modeling for Anomaly Detection. ICLR (2024).
4. Watson, D. et al. FraudDiffuse: Diffusion-aided Synthetic Fraud Augmentation for Improved Fraud Detection. ACM ICAIF (2024).
5. Lopez-Rojas, E., Elmir, A., Axelsson, S. PaySim: a financial mobile money simulator for fraud detection. EMSS (2016).
6. Jesus, S. et al. Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation. NeurIPS Datasets and Benchmarks (2022).