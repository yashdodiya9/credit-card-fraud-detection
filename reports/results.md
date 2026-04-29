Split: **time** | train 170,884 / val 56,961 / test 56,962 transactions | frauds in test: **75** | assumed review cost per alert: **5**

| Model | Val PR-AUC | Test PR-AUC | 95% CI | Test ROC-AUC | Precision | Recall | Alerts/10k txns | Fraud value caught % | Cost saving vs no model % |
|---|---|---|---|---|---|---|---|---|---|
| logreg_balanced | 0.757 | 0.748 | 0.629-0.853 | 0.981 | 0.96 | 0.64 | 8.8 | 60.3 | 57.0 |
| random_forest_balanced | 0.761 | 0.807 | 0.710-0.885 | 0.979 | 0.85 | 0.76 | 11.8 | 65.9 | 61.5 |
| xgboost_weighted | 0.784 | 0.806 | 0.711-0.881 | 0.972 | 0.92 | 0.72 | 10.4 | 52.4 | 48.6 |
| logreg_smote | 0.704 | 0.741 | 0.631-0.826 | 0.977 | 0.78 | 0.67 | 11.2 | 60.4 | 56.3 |
| xgboost_smote | 0.758 | 0.798 | 0.703-0.873 | 0.972 | 0.91 | 0.69 | 10.0 | 51.9 | 48.2 |

Selected model (best validation cost saving): **xgboost_smote**. Thresholds are tuned on the validation set only; all numbers above are on the held-out test set. The test set has few fraud cases, so read the confidence interval, not just the point estimate.

![PR curves](reports/figures/pr_curves.png)
