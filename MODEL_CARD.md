# Model Card — Telco Churn Logistic Regression

## Intended use
Educational and portfolio analysis of customer churn prediction and interpretable risk segmentation.

## Model
Logistic Regression with:
- median imputation for numerical features
- most-frequent imputation for categorical features
- standard scaling for numerical features
- one-hot encoding for categorical features
- five-fold stratified cross-validation
- hyperparameter tuning over regularization strength C

## Validation snapshot
On the supplied dataset with a stratified 80/20 split:
- Best CV ROC-AUC: 0.8477
- Holdout ROC-AUC: 0.8439
- Holdout PR-AUC: 0.6462
- Holdout F1 at threshold 0.50: 0.5800

These values are reproducibility results for this project setup, not guarantees for future or external data.

## Limitations
The data are observational. Associations should not be treated as causal effects. The model should not be used for automated customer decisions without business validation, monitoring, fairness review, cost analysis, and appropriate governance.
