"""Train/val/test splits: random and (balanced) Bemis-Murcko scaffold splits."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def random_split(n: int, seed: int, frac=(0.8, 0.1, 0.1)):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train, n_val = int(frac[0] * n), int(frac[1] * n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def scaffold_split(smiles: list[str], seed: int, frac=(0.8, 0.1, 0.1)):
    """Balanced scaffold split.

    Molecules sharing a scaffold always land in the same set, so test molecules
    are structurally distinct from training molecules. Scaffold groups larger
    than half the validation/test size go to train first; the remaining groups
    are shuffled with ``seed`` so different seeds give different splits.
    """
    groups = defaultdict(list)
    for i, smi in enumerate(smiles):
        groups[murcko_scaffold(smi)].append(i)

    n = len(smiles)
    n_train, n_val = int(frac[0] * n), int(frac[1] * n)
    n_test = n - n_train - n_val
    big_cutoff = n_val / 2

    big = [g for g in groups.values() if len(g) > big_cutoff]
    small = [g for g in groups.values() if len(g) <= big_cutoff]
    rng = np.random.RandomState(seed)
    rng.shuffle(big)
    rng.shuffle(small)

    train, val, test = [], [], []
    for group in big + small:
        if len(train) + len(group) <= n_train:
            train += group
        elif len(val) + len(group) <= n_val:
            val += group
        else:
            test += group
    assert len(train) + len(val) + len(test) == n
    return np.array(train), np.array(val), np.array(test)


def scaffold_overlap(smiles: list[str], train_idx, test_idx) -> float:
    """Fraction of test molecules whose scaffold also appears in train."""
    train_scaf = {murcko_scaffold(smiles[i]) for i in train_idx}
    hits = sum(murcko_scaffold(smiles[i]) in train_scaf for i in test_idx)
    return hits / max(len(test_idx), 1)
