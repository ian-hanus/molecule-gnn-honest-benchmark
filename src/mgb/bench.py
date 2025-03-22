"""Benchmark harness: one (dataset, split, seed, model) run -> a metrics row."""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

from . import baselines, metrics
from .data import load
from .featurize import ecfp_matrix
from .splits import random_split, scaffold_overlap, scaffold_split


@lru_cache(maxsize=None)
def _dataset(name: str):
    df, spec = load(name)
    smiles = df["smiles"].tolist()
    Y = df[spec.task_cols].to_numpy(dtype=np.float64)
    return smiles, Y, spec


@lru_cache(maxsize=None)
def _fingerprints(name: str) -> np.ndarray:
    smiles, _, _ = _dataset(name)
    return ecfp_matrix(smiles)


def make_split(name: str, split: str, seed: int):
    smiles, _, _ = _dataset(name)
    if split == "scaffold":
        return scaffold_split(smiles, seed)
    if split == "random":
        return random_split(len(smiles), seed)
    raise ValueError(f"unknown split: {split}")


def score(spec, Y_test: np.ndarray, pred: dict) -> dict:
    if spec.task_type == "regression":
        y = Y_test[:, 0]
        out = {"rmse": metrics.rmse(y, pred["mu"]), "mae": metrics.mae(y, pred["mu"])}
        if "sigma" in pred:
            out["coverage90"] = metrics.interval_coverage(y, pred["mu"], pred["sigma"])
            out["nll"] = metrics.gaussian_nll(y, pred["mu"], pred["sigma"])
        return out
    return {
        "auc": metrics.multitask_auc(Y_test, pred["prob"]),
        "ece": metrics.ece(Y_test, pred["prob"]),
    }


def run_baseline(model: str, dataset: str, split: str, seed: int) -> dict:
    smiles, Y, spec = _dataset(dataset)
    X = _fingerprints(dataset)
    tr, va, te = make_split(dataset, split, seed)
    t0 = time.time()
    pred = baselines.fit_predict(model, spec.task_type, X[tr], Y[tr], X[te], seed)
    row = {
        "dataset": dataset, "split": split, "seed": seed, "model": model,
        "n_train": len(tr), "n_test": len(te),
        "scaffold_overlap": scaffold_overlap(smiles, tr, te),
        "seconds": round(time.time() - t0, 1),
    }
    row.update(score(spec, Y[te], pred))
    return row
