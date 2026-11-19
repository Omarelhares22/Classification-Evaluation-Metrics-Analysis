# Classification Metrics Analysis

IEEE PUA · AI & Academic Learning Track (Level 2) — Practical Task, Part 2.

Evaluates a Logistic Regression classifier on the Wisconsin Breast Cancer Diagnostic dataset (positive class = malignant). The notebook is self-contained: it rebuilds the same stratified split, scaler and model (seed 42) used in `logistic_regression_pipeline`.

## Project structure
```
classification_metrics_analysis/
├── notebook_2.ipynb      # full analysis (executed, outputs included)
├── requirements.txt
├── README.md
└── figures/              # metric visual outputs
    ├── confusion_matrix.png
    ├── roc_auc_curve.png
    ├── threshold_confusion_matrices.png
    └── precision_recall_vs_threshold.png
```

## Run
```bash
pip install -r requirements.txt
jupyter notebook notebook_2.ipynb
```

## Results (test set, n = 114)
| Threshold | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| 0.3 | 41 | 1 | 1 | 71 | 0.9825 | 0.9762 | 0.9762 | 0.9762 |
| **0.5** | 39 | 1 | 3 | 71 | 0.9649 | 0.9750 | 0.9286 | 0.9512 |
| 0.7 | 38 | 0 | 4 | 72 | 0.9649 | 1.0000 | 0.9048 | 0.9500 |

- **AUC = 0.9960** (0.5 = random, 1.0 = perfect).
- Manual Precision / Recall / F1 / Accuracy match `sklearn` and `classification_report` exactly (comparison table in the notebook).

## Key takeaways
1. At the default 0.5 threshold the model is highly precise (0.975) but misses 3 of 42 malignant cases (recall 0.929).
2. Lowering the threshold to 0.3 trades a little probability confidence for far fewer missed cancers (FN 3 → 1) with no extra false positives here.
3. Raising it to 0.7 eliminates false alarms (precision 1.0) at the cost of recall (0.905).
4. For medical screening, recall usually matters more than precision, so a lower threshold is preferable.
5. The test set is small; one or two samples change the metrics noticeably.
