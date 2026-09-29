# Telco Customer Churn — Logistic Regression

An end-to-end, business-focused customer churn project built around **interpretable Logistic Regression**.

## Results at a glance

| Metric | Result |
|---|---:|
| 5-fold CV ROC-AUC | **0.8477** |
| Holdout ROC-AUC | **0.8439** |
| Holdout PR-AUC | **0.6462** |
| Holdout F1 @ 0.50 | **0.5800** |

These are reproducibility results for the supplied dataset and project configuration.

## Master Question

> Which customer characteristics, service patterns, contract structures, and billing behaviors are associated with customer churn; how accurately can Logistic Regression estimate churn probability; and how can those predictions be translated into interpretable customer-risk segments for business analysis?

## Methodology

```text
Raw CSV
  -> validation and cleaning
  -> business feature engineering
  -> stratified 80/20 split
  -> leakage-safe preprocessing
  -> 5-fold ROC-AUC cross-validation
  -> Logistic Regression C tuning
  -> holdout evaluation
  -> threshold analysis
  -> coefficient / odds-ratio interpretation
  -> customer risk segmentation
```

## What this project covers

- Business problem definition and dataset validation
- Missing, blank, duplicate, type, and consistency checks
- Exploratory and segmentation analysis
- Leakage-safe imputation, scaling, and one-hot encoding
- Logistic Regression with L1/L2 support and class weighting
- 5-fold cross-validation and regularization tuning
- Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Log Loss, and Brier Score
- Confusion matrix and probability-threshold analysis
- Coefficients, odds ratios, probability interpretation, and risk segments
- Robustness, limitations, and business-impact discussion

## Dataset

The repository includes:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains **7,043 rows and 21 columns** with `Churn` as the binary target.

## Project structure

```text
telco-customer-churn-logistic-regression/
├── data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/telco_customer_churn_logistic_regression.ipynb
├── src/
│   ├── analysis.py
│   └── train_model.py
├── tests/test_analysis.py
├── docs/
│   ├── PROJECT_QUESTIONS.md
│   └── Telco_Customer_Churn_Project_Questions.docx
├── .github/workflows/ci.yml
├── MODEL_CARD.md
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

Open the notebook at `notebooks/telco_customer_churn_logistic_regression.ipynb`.

### Run tests

```bash
pytest -q
```

### Run tests with coverage

```bash
pytest --cov=src --cov-report=term-missing
```

### Run linting

```bash
ruff check .
```

### Train the model

```bash
python src/train_model.py
```

The training script performs a stratified holdout split, five-fold ROC-AUC tuning over Logistic Regression `C`, evaluates the untouched holdout set, and saves the fitted pipeline to `models/logistic_regression_pipeline.joblib`.

## Quality controls

GitHub Actions runs:

1. Ruff linting
2. Pytest with an 85% coverage gate
3. Python compilation checks

## Analytical principle

Observed associations should not automatically be interpreted as causal effects. The dataset supports predictive and descriptive analysis; it does not establish that changing a feature will cause churn to increase or decrease.

## Reproducibility

The raw CSV is committed to the repository for the project. Verify the original dataset's licensing/source terms before independently redistributing the dataset.

## Author

**Alok Agarwal**  
Data Science • Python • SQL • Machine Learning • Analytics
