"""Run the fingerprint baselines over datasets x splits x seeds."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mgb.bench import run_baseline  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+", default=["esol", "bbbp", "tox21"])
parser.add_argument("--models", nargs="+", default=["rf", "xgb"])
parser.add_argument("--splits", nargs="+", default=["random", "scaffold"])
parser.add_argument("--seeds", type=int, default=5)
parser.add_argument("--out", default="results/baselines.csv")
args = parser.parse_args()

rows = []
for dataset in args.datasets:
    for split in args.splits:
        for model in args.models:
            for seed in range(args.seeds):
                row = run_baseline(model, dataset, split, seed)
                rows.append(row)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f"wrote {out}")
