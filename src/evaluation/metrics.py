"""Evaluation metrics for anomaly scores vs binary fraud labels.

Matches the evaluation protocol described in the proposal: AUROC, Average
Precision, and F1 at 1% FPR, all computed from a continuous anomaly score
(higher = more anomalous / more likely fraud) against ground-truth labels
(1 = fraud, 0 = normal). None of these need a chosen decision threshold
except the F1@1%FPR metric, which picks its own threshold internally.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve. 0.5 = random, 1.0 = perfect separation."""
    return float(roc_auc_score(y_true, scores))


def compute_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve. More informative than AUROC
    under heavy class imbalance (fraud is ~0.1-0.2% of transactions here),
    since it doesn't get inflated by the huge number of true negatives.
    """
    return float(average_precision_score(y_true, scores))


def compute_f1_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """F1 score at the operating threshold where the false positive rate
    is (approximately) `target_fpr`. This simulates a realistic deployment
    constraint: a fraud team can only manually review a small fraction of
    the (overwhelmingly normal) transactions flagged, so we fix the false
    alarm budget and ask how much fraud gets caught at that budget.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    # roc_curve returns fpr in increasing order; find the largest threshold
    # whose fpr does not exceed the target (i.e. the strictest threshold
    # that still respects the false-positive budget).
    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) == 0:
        # target_fpr smaller than the smallest achievable fpr - fall back
        # to the strictest available threshold.
        idx = 0
    else:
        idx = valid_idx[-1]
    threshold = thresholds[idx]
    y_pred = (scores >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


def evaluate_all(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> Dict[str, float]:
    """Convenience wrapper: runs every metric and returns a name -> value dict."""
    return {
        "auroc": compute_auroc(y_true, scores),
        "average_precision": compute_average_precision(y_true, scores),
        f"f1_at_{int(target_fpr * 100)}pct_fpr": compute_f1_at_fpr(y_true, scores, target_fpr),
    }
