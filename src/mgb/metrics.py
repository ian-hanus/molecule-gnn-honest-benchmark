"""Evaluation metrics, including calibration.

Multi-task labels use NaN for "not measured"; every metric masks those out.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def multitask_auc(y: np.ndarray, prob: np.ndarray) -> float:
    """Mean ROC-AUC over tasks that have both classes among measured labels."""
    y, prob = np.atleast_2d(y.T).T, np.atleast_2d(prob.T).T
    aucs = []
    for t in range(y.shape[1]):
        m = ~np.isnan(y[:, t])
        if m.sum() and len(np.unique(y[m, t])) == 2:
            aucs.append(roc_auc_score(y[m, t], prob[m, t]))
    return float(np.mean(aucs)) if aucs else float("nan")


def ece(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error for binary predictions, equal-width bins."""
    y, prob = np.asarray(y).ravel(), np.asarray(prob).ravel()
    m = ~np.isnan(y)
    y, prob = y[m], prob[m]
    bins = np.minimum((prob * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        sel = bins == b
        if sel.any():
            total += sel.mean() * abs(y[sel].mean() - prob[sel].mean())
    return float(total)


def reliability_curve(y: np.ndarray, prob: np.ndarray, n_bins: int = 10):
    """Return (mean predicted prob, observed frequency, count) per non-empty bin."""
    y, prob = np.asarray(y).ravel(), np.asarray(prob).ravel()
    m = ~np.isnan(y)
    y, prob = y[m], prob[m]
    bins = np.minimum((prob * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        sel = bins == b
        if sel.any():
            out.append((float(prob[sel].mean()), float(y[sel].mean()), int(sel.sum())))
    return out


def interval_coverage(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, z: float = 1.645) -> float:
    """Fraction of targets inside mu +/- z*sigma (z=1.645 is a nominal 90% interval)."""
    return float(np.mean(np.abs(y - mu) <= z * sigma))


def gaussian_nll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, eps: float = 1e-6) -> float:
    var = np.maximum(sigma, eps) ** 2
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (y - mu) ** 2 / var))
