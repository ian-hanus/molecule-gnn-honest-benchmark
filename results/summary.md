# Results

## Headline numbers

### ESOL: RMSE (lower is better), mean ± std over 5 splits

| model | random split | scaffold split | scaffold − random |
|---|---|---|---|
| rf | 0.861 ± 0.053 | 1.147 ± 0.047 | +0.286 |
| xgb | 0.745 ± 0.083 | 1.029 ± 0.053 | +0.285 |
| gcn | 0.737 ± 0.083 | 1.053 ± 0.092 | +0.316 |
| gin | 0.726 ± 0.096 | 1.028 ± 0.075 | +0.302 |
| dmpnn | 0.686 ± 0.043 | 1.044 ± 0.124 | +0.358 |
| gcn-ens3 | 0.728 ± 0.077 | 1.012 ± 0.108 | +0.284 |
| gin-ens3 | 0.694 ± 0.062 | 0.973 ± 0.087 | +0.279 |
| dmpnn-ens3 | 0.656 ± 0.038 | 0.998 ± 0.121 | +0.343 |

### BBBP: ROC-AUC (higher is better), mean ± std over 5 splits

| model | random split | scaffold split | scaffold − random |
|---|---|---|---|
| rf | 0.930 ± 0.020 | 0.901 ± 0.026 | -0.028 |
| xgb | 0.919 ± 0.017 | 0.892 ± 0.037 | -0.027 |
| gcn | 0.921 ± 0.021 | 0.875 ± 0.027 | -0.046 |
| gin | 0.897 ± 0.031 | 0.873 ± 0.024 | -0.024 |
| dmpnn | 0.919 ± 0.014 | 0.870 ± 0.048 | -0.049 |
| gcn-ens3 | 0.927 ± 0.022 | 0.883 ± 0.024 | -0.044 |
| gin-ens3 | 0.918 ± 0.012 | 0.869 ± 0.024 | -0.049 |
| dmpnn-ens3 | 0.922 ± 0.013 | 0.875 ± 0.041 | -0.047 |

### TOX21: mean ROC-AUC (higher is better), mean ± std over 3 splits

| model | random split | scaffold split | scaffold − random |
|---|---|---|---|
| rf | 0.828 ± 0.006 | 0.802 ± 0.025 | -0.026 |
| xgb | 0.802 ± 0.007 | 0.775 ± 0.015 | -0.027 |
| gcn | 0.834 ± 0.009 | 0.806 ± 0.002 | -0.029 |
| gin | 0.833 ± 0.014 | 0.808 ± 0.010 | -0.025 |
| dmpnn | 0.820 ± 0.009 | 0.796 ± 0.016 | -0.024 |
| gcn-ens3 | 0.836 ± 0.008 | 0.814 ± 0.006 | -0.022 |
| gin-ens3 | 0.842 ± 0.014 | 0.821 ± 0.006 | -0.021 |
| dmpnn-ens3 | 0.828 ± 0.013 | 0.803 ± 0.014 | -0.024 |


## Scaffold leakage of each split type

Share of test molecules whose Bemis-Murcko scaffold also appears in train.

| dataset | random split | scaffold split |
|---|---|---|
| bbbp | 58% | 0% |
| esol | 81% | 0% |
| tox21 | 77% | 0% |

## Calibration

### ESOL, scaffold split: does the ensemble's spread give honest intervals?

Nominal coverage of the mu ± 1.645σ interval is 90%.

| model | 90% interval coverage | RMSE |
|---|---|---|
| rf | 0.91 ± 0.03 | 1.147 ± 0.047 |
| gcn-ens3 | 0.32 ± 0.07 | 1.012 ± 0.108 |
| gin-ens3 | 0.38 ± 0.07 | 0.973 ± 0.087 |
| dmpnn-ens3 | 0.25 ± 0.09 | 0.998 ± 0.121 |

### Classification calibration (ECE, lower is better), scaffold split

| model | BBBP | Tox21 |
|---|---|---|
| rf | 0.071 ± 0.027 | 0.016 ± 0.006 |
| xgb | 0.073 ± 0.019 | 0.115 ± 0.013 |
| gcn | 0.087 ± 0.020 | 0.015 ± 0.006 |
| gin | 0.102 ± 0.021 | 0.018 ± 0.005 |
| dmpnn | 0.098 ± 0.025 | 0.016 ± 0.008 |
| gcn-ens3 | 0.080 ± 0.022 | 0.011 ± 0.007 |
| gin-ens3 | 0.088 ± 0.016 | 0.014 ± 0.004 |
| dmpnn-ens3 | 0.096 ± 0.021 | 0.015 ± 0.003 |
