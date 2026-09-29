# Model Card — Telco Churn Logistic Regression

## Intended use

Educational and portfolio analysis of customer churn prediction and interpretable risk segmentation.

## Model

Logistic Regression with:

- median imputation for numerical features
- most-frequent imputation for categorical features
- standard scaling for numerical features
- one-hot encoding for categorical features
- stratified 80/20 train/holdout split
- five-fold ROC-AUC cross-validation
- regularization-strength tuning over candidate C values
- probability-based threshold analysis
- coefficient and odds-ratio interpretation

## Validation snapshot

| Metric | Result |
|---|---:|
| Best CV ROC-AUC | 0.8477 |
| Holdout ROC-AUC | 0.8439 |
| Holdout PR-AUC | 0.6462 |
| Holdout F1 @ 0.50 | 0.5800 |

These are reproducibility results for this project setup, not guarantees for future or external data.

## Limitations

The data are observational. Associations should not be treated as causal effects.

The model should not be used for automated customer decisions without business validation, monitoring, fairness review, cost analysis, and appropriate governance.

Performance may change with population drift, data-quality changes, feature-definition changes, or different sampling procedures.
