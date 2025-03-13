"""Molecule featurization: ECFP fingerprints and molecular graphs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

ATOMS = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C N O F P S Cl Br I, everything else -> "other"
HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]


def _one_hot(value, choices, with_other=True):
    vec = [float(value == c) for c in choices]
    if with_other:
        vec.append(float(value not in choices))
    return vec


def atom_features(atom: Chem.Atom) -> list[float]:
    return (
        _one_hot(atom.GetAtomicNum(), ATOMS)
        + _one_hot(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
        + _one_hot(atom.GetFormalCharge(), [-1, 0, 1])
        + _one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    return _one_hot(bond.GetBondType(), BOND_TYPES, with_other=False) + [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


ATOM_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
BOND_DIM = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


def ecfp(smiles: str, radius: int = 2, n_bits: int = 2048, counts: bool = True) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = gen.GetCountFingerprintAsNumPy(mol) if counts else gen.GetFingerprintAsNumPy(mol)
    return fp.astype(np.float32)


def ecfp_matrix(smiles: list[str], **kwargs) -> np.ndarray:
    return np.stack([ecfp(s, **kwargs) for s in smiles])


@dataclass
class MolGraph:
    x: torch.Tensor           # [n_atoms, ATOM_DIM]
    edge_index: torch.Tensor  # [2, n_edges], both directions
    edge_attr: torch.Tensor   # [n_edges, BOND_DIM]


def mol_to_graph(smiles: str) -> MolGraph | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    src, dst, attr = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        src += [i, j]
        dst += [j, i]
        attr += [f, f]
    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(attr, dtype=torch.float32)
    else:  # single-atom molecules
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_DIM), dtype=torch.float32)
    return MolGraph(x, edge_index, edge_attr)


@dataclass
class GraphBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    batch: torch.Tensor  # molecule id of each atom
    n_graphs: int

    def to(self, device) -> "GraphBatch":
        return GraphBatch(
            self.x.to(device), self.edge_index.to(device),
            self.edge_attr.to(device), self.batch.to(device), self.n_graphs,
        )


def collate(graphs: list[MolGraph]) -> GraphBatch:
    xs, eis, eas, batch = [], [], [], []
    offset = 0
    for gi, g in enumerate(graphs):
        xs.append(g.x)
        eis.append(g.edge_index + offset)
        eas.append(g.edge_attr)
        batch.append(torch.full((g.x.shape[0],), gi, dtype=torch.long))
        offset += g.x.shape[0]
    return GraphBatch(
        torch.cat(xs), torch.cat(eis, dim=1), torch.cat(eas),
        torch.cat(batch), len(graphs),
    )
