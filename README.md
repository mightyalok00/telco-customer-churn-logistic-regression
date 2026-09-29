# Telco Customer Churn — Logistic Regression

An end-to-end, business-focused customer churn project built around **interpretable Logistic Regression**.

## Master Question

> Which customer characteristics, service patterns, contract structures, and billing behaviors are associated with customer churn; how accurately can Logistic Regression estimate churn probability; and how can those predictions be translated into interpretable customer-risk segments for business analysis?

## What this project covers

- Business problem definition
- Dataset structure and target analysis
- Missing, blank, duplicate, type, and consistency validation
- Exploratory data analysis
- Deep business segmentation
- Statistical association analysis
- Leakage-safe preprocessing
- One-hot encoding and numerical scaling
- Logistic Regression
- L1/L2 regularization
- Class weighting
- 5-fold cross-validation
- Hyperparameter tuning
- Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Log Loss, Brier Score
- Confusion matrix and threshold analysis
- Probability calibration
- Coefficient and odds-ratio interpretation
- Customer risk segmentation
- Decision Tree / Random Forest comparison
- Robustness and business-impact analysis

## Dataset

Place the supplied CSV at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The included source dataset contains 7,043 customer records and 21 columns. `Churn` is the binary target.

## Project structure

```text
telco-customer-churn-logistic-regression/
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── telco_customer_churn_logistic_regression.ipynb
├── src/
│   ├── __init__.py
│   ├── analysis.py
│   └── train_model.py
├── tests/
│   └── test_analysis.py
├── reports/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

Open:

```text
notebooks/telco_customer_churn_logistic_regression.ipynb
```

## Run tests

```bash
pytest -q
```

## Train from the command line

```bash
python src/train_model.py
```

The training script performs a stratified train/test split, five-fold ROC-AUC grid search over Logistic Regression `C`, and saves the fitted pipeline under `models/`.

## Important analytical principle

Observed associations in this project should not automatically be interpreted as causal effects. The dataset supports predictive and descriptive analysis; it does not by itself establish that changing a feature will cause churn to increase or decrease.

## Suggested GitHub repository name

`telco-customer-churn-logistic-regression`

## Author

**Alok Agarwal**  
Data Science • Python • SQL • Machine Learning • Analytics
