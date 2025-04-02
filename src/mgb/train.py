"""Training loop and seed ensembles for the molecule GNNs."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .featurize import MolGraph, collate
from .models import MoleculeGNN


@dataclass
class TrainConfig:
    dim: int = 128
    depth: int = 3
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 100
    patience: int = 20


def configure_threads(n: int | None = None) -> None:
    """Cap intra-op threads. These graphs are tiny, and using every core (including
    efficiency cores) was ~50x slower than 4 threads when profiled on an M-series Mac."""
    torch.set_num_threads(n or min(4, os.cpu_count() or 1))


def _loss(task_type: str, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Masked loss: NaN labels (unmeasured tasks) contribute nothing."""
    mask = ~torch.isnan(y)
    y = torch.nan_to_num(y)
    if task_type == "regression":
        per = (out - y) ** 2
    else:
        per = nn.functional.binary_cross_entropy_with_logits(out, y, reduction="none")
    return (per * mask).sum() / mask.sum().clamp(min=1)


def _iter_batches(graphs, Y, idx, batch_size, rng=None):
    idx = np.asarray(idx)
    if rng is not None:
        idx = idx[rng.permutation(len(idx))]
    for start in range(0, len(idx), batch_size):
        sel = idx[start:start + batch_size]
        yield collate([graphs[i] for i in sel]), torch.tensor(Y[sel], dtype=torch.float32)


@torch.no_grad()
def _predict_raw(model, graphs, Y, idx, device, batch_size=256) -> np.ndarray:
    model.eval()
    outs = [model(g.to(device)).cpu() for g, _ in _iter_batches(graphs, Y, idx, batch_size)]
    return torch.cat(outs).numpy()


def train_one(
    arch: str,
    task_type: str,
    graphs: list[MolGraph],
    Y: np.ndarray,
    train_idx,
    val_idx,
    test_idx,
    seed: int,
    cfg: TrainConfig = TrainConfig(),
    device: str = "cpu",
) -> dict:
    """Train one model with early stopping on validation loss; predict the test set."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    if task_type == "regression":
        mean, std = np.nanmean(Y[train_idx]), np.nanstd(Y[train_idx]) + 1e-8
        Yn = (Y - mean) / std
    else:
        Yn = Y

    model = MoleculeGNN(arch, Y.shape[1], cfg.dim, cfg.depth, cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val, best_state, stale = float("inf"), None, 0
    for _ in range(cfg.epochs):
        model.train()
        for g, y in _iter_batches(graphs, Yn, train_idx, cfg.batch_size, rng):
            if g.n_graphs < 2:  # BatchNorm needs more than one graph
                continue
            opt.zero_grad()
            _loss(task_type, model(g.to(device)), y.to(device)).backward()
            opt.step()

        val_out = torch.tensor(_predict_raw(model, graphs, Yn, val_idx, device))
        val_loss = _loss(task_type, val_out, torch.tensor(Yn[val_idx], dtype=torch.float32)).item()
        if val_loss < best_val - 1e-5:
            best_val, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    model.load_state_dict(best_state)
    out = _predict_raw(model, graphs, Yn, test_idx, device)
    if task_type == "regression":
        return {"mu": out[:, 0] * std + mean}
    return {"prob": 1.0 / (1.0 + np.exp(-out))}


def ensemble(members: list[dict]) -> dict:
    """Combine member predictions: mean, plus std across members for regression."""
    if "mu" in members[0]:
        mus = np.stack([m["mu"] for m in members])
        return {"mu": mus.mean(0), "sigma": mus.std(0)}
    return {"prob": np.stack([m["prob"] for m in members]).mean(0)}
