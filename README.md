# molecule-gnn-honest-benchmark

Do graph neural networks beat fingerprint baselines on small molecular property
datasets, once the split and the uncertainty are handled honestly?

**Short answer:** not reliably. On three MoleculeNet datasets the differences between
GNNs and random forest / XGBoost on ECFP counts are mostly smaller than the
split-to-split noise. What does change the picture is the split: scaffold splitting
costs every model about the same amount. Ensemble spread is a poor uncertainty
estimate on out-of-scaffold molecules.

![scaffold gap](results/fig_scaffold_gap.png)

## Setup

- **Data:** MoleculeNet ESOL (regression, RMSE), BBBP (binary, ROC-AUC) and Tox21
  (12 tasks with missing labels, mean masked ROC-AUC). Molecules RDKit cannot parse are
  dropped so every model sees the same rows.
- **Splits:** a random 80/10/10 split and a balanced Bemis-Murcko **scaffold split**
  (whole scaffolds go to one side). 5 split seeds for ESOL and BBBP, 3 for Tox21.
  A test checks that no scaffold crosses the scaffold split.
- **Baselines:** random forest and XGBoost on ECFP counts.
- **GNNs:** plain-PyTorch GCN, GINE and D-MPNN, written without PyG. Each is trained with
  early stopping on a validation set (60 epochs, patience 12) and with masked multi-task
  losses for Tox21. Each GNN is also run as a 3-member seed ensemble (`-ens3`).
- **No hyperparameter tuning** for any model. Settings were fixed in advance, so a tuned
  baseline or a tuned GNN could move the numbers.

## Results

Mean ± std over split seeds. Full tables are in [`results/summary.md`](results/summary.md).

### Scaffold split (the harder, more realistic test)

| model | ESOL RMSE ↓ | BBBP AUC ↑ | Tox21 AUC ↑ |
|---|---|---|---|
| rf | 1.147 ± 0.047 | **0.901 ± 0.026** | 0.802 ± 0.025 |
| xgb | 1.029 ± 0.053 | 0.892 ± 0.037 | 0.775 ± 0.015 |
| gcn | 1.053 ± 0.092 | 0.875 ± 0.027 | 0.806 ± 0.002 |
| gin | 1.028 ± 0.075 | 0.873 ± 0.024 | 0.808 ± 0.010 |
| dmpnn | 1.044 ± 0.124 | 0.870 ± 0.048 | 0.796 ± 0.016 |
| gcn-ens3 | 1.012 ± 0.108 | 0.883 ± 0.024 | 0.814 ± 0.006 |
| gin-ens3 | **0.973 ± 0.087** | 0.869 ± 0.024 | **0.821 ± 0.006** |
| dmpnn-ens3 | 0.998 ± 0.121 | 0.875 ± 0.041 | 0.803 ± 0.014 |

### What I take from this

1. **No clear GNN win.** On ESOL the best model (gin-ens3, 0.973) is within one standard
   deviation of XGBoost (1.029). On BBBP the random forest is the best model and every GNN is
   about 0.02 to 0.03 AUC behind it. Only on Tox21 do GNNs sit slightly above the baselines
   (best 0.821 vs 0.802 for RF), and that is 3 seeds with std up to 0.025.
2. **Ensembling helps a little, usually.** An `-ens3` model beats its single-model version in
   most cells (ESOL scaffold RMSE improves by 0.04 to 0.055), but not all: on BBBP scaffold
   gin-ens3 is slightly worse than gin (0.869 vs 0.873). The gains are comparable to the noise.
3. **Random splits flatter every model.** Test molecules share a scaffold with the training
   set 58% (BBBP), 77% (Tox21) and 81% (ESOL) of the time under a random split, versus 0% by
   construction under a scaffold split. Moving to the scaffold split raises ESOL RMSE by about
   0.28 to 0.36 for every model and lowers classification AUC by about 0.02 to 0.05.
   The size of that penalty is similar across model families, so in this data I could not
   show that GNNs are more or less hurt by scaffold shift than fingerprints.
4. **Ensemble spread is not an honest uncertainty estimate here.** On the ESOL scaffold
   split, a nominal 90% interval (mean ± 1.645 sigma of the ensemble) covers the truth only
   25% to 38% of the time for the GNN ensembles. The random forest's per-tree spread covers
   91%, although its RMSE is the worst of the group. With only 3 members the ensembles are
   overconfident; I did not try recalibration.
5. **Classification calibration is mostly fine**, with ECE around 0.01 to 0.02 on Tox21 for
   everything except XGBoost (0.115), and about 0.07 to 0.10 on BBBP, where the random forest
   and XGBoost are slightly better calibrated than the GNNs.

## Limitations

- **Few seeds** (5 for ESOL and BBBP, 3 for Tox21). Most gaps between models are inside
  one standard deviation and should be read as "no detectable difference".
- **No tuning** for any model, and a short training schedule (60 epochs, patience 12).
  Tuned GNNs, pretrained GNNs or larger ensembles could change the ranking.
- **Small ensembles** (3 members), which is probably why the interval coverage is so low.
- **The interval check is ESOL only.** I did not evaluate uncertainty on the classification
  datasets beyond ECE.
- **Scaffold splits are one kind of shift.** I did not do the error-versus-similarity analysis
  (accuracy as a function of distance to the nearest training molecule) that would show where
  the models actually fail.
- Three datasets of modest size, so nothing here says how the methods compare on large or
  3D-dependent benchmarks.

## Reproduce

```bash
conda env create -f environment.yml && conda activate bioml
export PYTHONPATH=src
python scripts/run_baselines.py        # RF / XGBoost
python scripts/run_gnns.py             # GCN / GINE / D-MPNN and ensembles (resumable)
python scripts/make_report.py          # tables and figure
pytest
```
