
"""Train and evaluate the final Telco churn Logistic Regression model."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

from analysis import (
    TARGET,
    load_data,
    clean_data,
    add_business_features,
    build_logistic_pipeline,
    evaluate_binary_model,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

data = add_business_features(clean_data(load_data(DATA)))
X = data.drop(columns=[TARGET])
y = (data[TARGET] == "Yes").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

pipeline = build_logistic_pipeline(X_train)
search = GridSearchCV(
    pipeline,
    param_grid={"model__C": [0.1, 0.5, 1.0, 2.0, 5.0]},
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
)
search.fit(X_train, y_train)

prob = search.predict_proba(X_test)[:, 1]
pred = (prob >= 0.50).astype(int)

metrics = evaluate_binary_model(y_test, pred, prob)
print("Best parameters:", search.best_params_)
print("Cross-validated ROC-AUC:", round(search.best_score_, 4))
print("Holdout metrics:", {k: round(v, 4) for k, v in metrics.items()})
print("\nClassification report:\n", classification_report(y_test, pred))

joblib.dump(search.best_estimator_, MODEL_DIR / "logistic_regression_pipeline.joblib")
print("Saved model to", MODEL_DIR / "logistic_regression_pipeline.joblib")
