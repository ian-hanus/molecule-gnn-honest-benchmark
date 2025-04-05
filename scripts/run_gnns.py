"""Run GNN ensembles over datasets x splits x seeds. Resumable: finished runs are skipped."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mgb.bench import run_gnn  # noqa: E402
from mgb.train import TrainConfig, configure_threads  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+", default=["esol", "bbbp", "tox21"])
parser.add_argument("--archs", nargs="+", default=["gcn", "gin", "dmpnn"])
parser.add_argument("--splits", nargs="+", default=["random", "scaffold"])
parser.add_argument("--seeds", type=int, default=5)
parser.add_argument("--members", type=int, default=3)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--patience", type=int, default=12)
parser.add_argument("--out", default="results/gnns.csv")
args = parser.parse_args()

FIELDS = [
    "dataset", "split", "seed", "model", "n_train", "n_test", "scaffold_overlap", "seconds",
    "rmse", "mae", "coverage90", "nll",  # regression
    "auc", "ece",                        # classification
]

configure_threads()
cfg = TrainConfig(epochs=args.epochs, patience=args.patience)
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)

done = set()
if out.exists():
    with out.open() as f:
        done = {(r["dataset"], r["split"], r["seed"], r["model"]) for r in csv.DictReader(f)}

for dataset in args.datasets:
    for split in args.splits:
        for arch in args.archs:
            for seed in range(args.seeds):
                if (dataset, split, str(seed), arch) in done:
                    continue
                rows = run_gnn(arch, dataset, split, seed, args.members, cfg)
                write_header = not out.exists()
                with out.open("a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS)
                    if write_header:
                        w.writeheader()
                    w.writerows(rows)
                print(dataset, split, arch, seed, {r["model"]: round(r.get("rmse", r.get("auc")), 3) for r in rows}, flush=True)
