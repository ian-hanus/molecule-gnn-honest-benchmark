"""Aggregate results/*.csv into results/summary.md and figures."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

PRIMARY = {"esol": ("rmse", "RMSE (lower is better)"),
           "bbbp": ("auc", "ROC-AUC (higher is better)"),
           "tox21": ("auc", "mean ROC-AUC (higher is better)")}
SINGLE = ["rf", "xgb", "gcn", "gin", "dmpnn"]
ORDER = SINGLE + ["gcn-ens3", "gin-ens3", "dmpnn-ens3"]


def load() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(RES.glob("baselines_*.csv")) + [RES / "gnns.csv"] if p.exists()]
    return pd.concat(frames, ignore_index=True)


def mean_std(s: pd.Series, digits: int = 3) -> str:
    return f"{s.mean():.{digits}f} ± {s.std(ddof=1):.{digits}f}" if len(s) > 1 else f"{s.mean():.{digits}f}"


def main_table(df: pd.DataFrame) -> str:
    lines = []
    for dataset, (metric, label) in PRIMARY.items():
        sub = df[df.dataset == dataset]
        if sub.empty:
            continue
        n_seeds = sub.seed.nunique()
        lines += [f"### {dataset.upper()}: {label}, mean ± std over {n_seeds} splits", "",
                  "| model | random split | scaffold split | scaffold − random |", "|---|---|---|---|"]
        for model in ORDER:
            r = sub[(sub.model == model) & (sub.split == "random")][metric]
            s = sub[(sub.model == model) & (sub.split == "scaffold")][metric]
            if r.empty or s.empty:
                continue
            lines.append(f"| {model} | {mean_std(r)} | {mean_std(s)} | {s.mean() - r.mean():+.3f} |")
        lines.append("")
    return "\n".join(lines)


def leakage_table(df: pd.DataFrame) -> str:
    sub = df[df.model == "rf"].groupby(["dataset", "split"]).scaffold_overlap.mean().unstack()
    lines = ["| dataset | random split | scaffold split |", "|---|---|---|"]
    for dataset, row in sub.iterrows():
        lines.append(f"| {dataset} | {row['random']:.0%} | {row['scaffold']:.0%} |")
    return "\n".join(lines)


def calibration_table(df: pd.DataFrame) -> str:
    lines = []
    esol = df[(df.dataset == "esol") & (df.split == "scaffold") & df.coverage90.notna()]
    if not esol.empty:
        lines += ["### ESOL, scaffold split: does the ensemble's spread give honest intervals?", "",
                  "Nominal coverage of the mu ± 1.645σ interval is 90%.", "",
                  "| model | 90% interval coverage | RMSE |", "|---|---|---|"]
        for model in ["rf", "gcn-ens3", "gin-ens3", "dmpnn-ens3"]:
            s = esol[esol.model == model]
            if not s.empty:
                lines.append(f"| {model} | {mean_std(s.coverage90, 2)} | {mean_std(s.rmse)} |")
        lines.append("")
    cls = df[(df.split == "scaffold") & df.ece.notna() & df.dataset.isin(["bbbp", "tox21"])]
    if not cls.empty:
        lines += ["### Classification calibration (ECE, lower is better), scaffold split", "",
                  "| model | BBBP | Tox21 |", "|---|---|---|"]
        for model in ORDER:
            cells = []
            for dataset in ["bbbp", "tox21"]:
                s = cls[(cls.model == model) & (cls.dataset == dataset)].ece
                cells.append(mean_std(s) if len(s) else "-")
            lines.append(f"| {model} | {cells[0]} | {cells[1]} |")
        lines.append("")
    return "\n".join(lines)


def gap_figure(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (dataset, (metric, label)) in zip(axes, PRIMARY.items()):
        sub = df[df.dataset == dataset]
        models = [m for m in SINGLE if m in set(sub.model)]
        x = np.arange(len(models))
        for off, split, color in [(-0.18, "random", "#9aa5b1"), (0.18, "scaffold", "#d1495b")]:
            vals = [sub[(sub.model == m) & (sub.split == split)][metric] for m in models]
            ax.bar(x + off, [v.mean() for v in vals], 0.34, yerr=[v.std(ddof=1) for v in vals],
                   color=color, label=split, capsize=2)
        ax.set_xticks(x, models)
        ax.set_title(f"{dataset.upper()}: {metric.upper()}")
        ax.set_ylabel(label.split(" (")[0])
        if metric == "auc":
            ax.set_ylim(0.5, 1.0)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    df = load()
    out = ["# Results", "", "## Headline numbers", "", main_table(df), "",
           "## Scaffold leakage of each split type", "",
           "Share of test molecules whose Bemis-Murcko scaffold also appears in train.", "",
           leakage_table(df), "", "## Calibration", "", calibration_table(df)]
    (RES / "summary.md").write_text("\n".join(out))
    gap_figure(df, RES / "fig_scaffold_gap.png")
    print((RES / "summary.md").read_text())
    sys.exit(0)
