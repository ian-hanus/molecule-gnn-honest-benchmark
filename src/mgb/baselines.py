"""Fingerprint baselines: random forest and XGBoost on ECFP counts.

Default hyperparameters on purpose: the point is a strong, cheap reference,
not a tuned one. Both models are fit one task at a time so multi-task labels
with missing values (Tox21) need no special handling.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor


def _fit_regression(name: str, X, y, X_test, seed: int):
    if name == "rf":
        model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)
        model.fit(X, y)
        per_tree = np.stack([t.predict(X_test) for t in model.estimators_])
        return {"mu": per_tree.mean(0), "sigma": per_tree.std(0)}
    model = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.5, tree_method="hist", n_jobs=-1, random_state=seed,
    )
    model.fit(X, y)
    return {"mu": model.predict(X_test)}


def _fit_classification_task(name: str, X, y, X_test, seed: int) -> np.ndarray:
    m = ~np.isnan(y)
    X, y = X[m], y[m]
    if len(np.unique(y)) < 2:  # degenerate task: fall back to the prior
        return np.full(len(X_test), float(y.mean()) if len(y) else 0.5)
    if name == "rf":
        model = RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=seed, class_weight="balanced_subsample",
        )
    else:
        pos = max(float((y == 1).sum()), 1.0)
        model = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.5, tree_method="hist", n_jobs=-1, random_state=seed,
            scale_pos_weight=float((y == 0).sum()) / pos,
        )
    model.fit(X, y.astype(int))
    return model.predict_proba(X_test)[:, 1]


def fit_predict(name: str, task_type: str, X_train, Y_train, X_test, seed: int = 0) -> dict:
    """Fit ``name`` ("rf" or "xgb") and predict ``X_test``.

    Returns ``{"mu", "sigma"?}`` for regression or ``{"prob"}`` for classification.
    """
    if task_type == "regression":
        return _fit_regression(name, X_train, Y_train[:, 0], X_test, seed)
    probs = [
        _fit_classification_task(name, X_train, Y_train[:, t], X_test, seed)
        for t in range(Y_train.shape[1])
    ]
    return {"prob": np.stack(probs, axis=1)}
