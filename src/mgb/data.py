"""Dataset loading for the MoleculeNet subsets used in this benchmark."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BASE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    smiles_col: str
    task_cols: list[str]
    task_type: str  # "regression" or "classification"


SPECS = {
    "esol": DatasetSpec(
        "esol", "delaney-processed.csv", "smiles",
        ["measured log solubility in mols per litre"], "regression",
    ),
    "bbbp": DatasetSpec("bbbp", "BBBP.csv", "smiles", ["p_np"], "classification"),
    "tox21": DatasetSpec("tox21", "tox21.csv.gz", "smiles", TOX21_TASKS, "classification"),
}


def download(name: str) -> Path:
    spec = SPECS[name]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / spec.filename
    if not path.exists():
        urllib.request.urlretrieve(BASE_URL + spec.filename, path)
    return path


def load(name: str) -> tuple[pd.DataFrame, DatasetSpec]:
    """Return a frame with a ``smiles`` column and one column per task."""
    spec = SPECS[name]
    df = pd.read_csv(download(name))
    df = df.rename(columns={spec.smiles_col: "smiles"})
    df = df[["smiles", *spec.task_cols]].dropna(subset=["smiles"])
    return df.reset_index(drop=True), spec
