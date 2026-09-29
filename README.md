# Telco Customer Churn — Logistic Regression

An end-to-end, business-focused customer churn project built around **interpretable Logistic Regression**, with a production-style **Streamlit analytics dashboard**.

## 🚀 Live application

Run the dashboard locally:

```bash
streamlit run app.py
```

The dashboard provides:

- Executive KPI overview
- Churn and contract analysis
- Interactive customer filtering
- Customer-level churn-risk prediction
- Adjustable classification threshold
- Precision / recall / F1 threshold analysis
- Holdout confusion matrix
- Logistic Regression coefficient and odds-ratio analysis
- Low / Medium / High customer-risk segmentation
- CSV export of model coefficients

The dashboard uses the same preprocessing and tuning methodology as the core project. It trains the model automatically on first load and caches the trained estimator for the session.

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
  -> Streamlit decision-support dashboard
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
- Interactive Streamlit exploration and prediction
- Robustness, limitations, and business-impact discussion

## Dataset

The repository includes:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains **7,043 rows and 21 columns** with `Churn` as the binary target.

## Project structure

```text
telco-customer-churn-logistic-regression/
├── app.py
├── .streamlit/config.toml
├── data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/telco_customer_churn_logistic_regression.ipynb
├── src/
│   ├── analysis.py
│   └── train_model.py
├── tests/
│   ├── test_analysis.py
│   └── test_train_model.py
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
```

### Launch the Streamlit dashboard

```bash
streamlit run app.py
```

### Run the notebook

```bash
jupyter notebook
```

Open `notebooks/telco_customer_churn_logistic_regression.ipynb`.

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

## Streamlit deployment

For Streamlit Community Cloud:

1. Push the repository to GitHub.
2. Create a new Streamlit app.
3. Select this repository.
4. Set the main file to `app.py`.
5. Deploy.

The dataset is included in the repository, so the dashboard does not require a separate data download.

## Quality controls

GitHub Actions runs:

1. Ruff linting
2. Pytest with an 85% coverage gate
3. Python and Streamlit-app compilation checks

The latest verified CI run reached **15 passing tests and 97.44% source coverage**.

## Analytical principle

Observed associations should not automatically be interpreted as causal effects. The dataset supports predictive and descriptive analysis; it does not establish that changing a feature will cause churn to increase or decrease.

The Streamlit predictor is intended for educational and portfolio demonstration. It should not be used as an automated customer decision system without business validation, monitoring, fairness review, cost analysis, and appropriate governance.

## Reproducibility

The raw CSV is committed to the repository for the project. Verify the original dataset's licensing/source terms before independently redistributing the dataset.

## Author

**Alok Agarwal**  
Data Science • Python • SQL • Machine Learning • Analytics
