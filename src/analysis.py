"""Reusable Telco Customer Churn analysis utilities."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20
DEFAULT_CV_FOLDS = 5
DEFAULT_C_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV and validate the required target column."""
    data = pd.read_csv(path)
    if TARGET not in data.columns:
        raise ValueError(f"Expected target column {TARGET!r}.")
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize TotalCharges, remove duplicates, and reset the index."""
    clean = data.copy()
    if "TotalCharges" in clean.columns:
        clean["TotalCharges"] = pd.to_numeric(clean["TotalCharges"], errors="coerce")
    return clean.drop_duplicates().reset_index(drop=True)


def add_business_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable customer-level features."""
    out = data.copy()
    service_cols = [
        "PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    present = [column for column in service_cols if column in out.columns]
    if present:
        out["ServiceCount"] = sum(
            out[column].isin(["Yes", "Yes - no internet service"]).astype(int)
            for column in present
        )
    if {"tenure", "TotalCharges", "MonthlyCharges"}.issubset(out.columns):
        out["AvgMonthlySpend"] = np.where(
            out["tenure"] > 0,
            out["TotalCharges"] / out["tenure"],
            out["MonthlyCharges"],
        )
    if "tenure" in out.columns:
        out["TenureBand"] = pd.cut(
            out["tenure"], bins=[-1, 6, 12, 24, 48, np.inf],
            labels=["0-6", "7-12", "13-24", "25-48", "49+"],
        )
    if "MonthlyCharges" in out.columns:
        out["ChargeBand"] = pd.qcut(
            out["MonthlyCharges"], q=4, duplicates="drop"
        ).astype(str)
    return out


def build_preprocessor(
    X: pd.DataFrame, drop_columns: Iterable[str] = (ID_COLUMN,)
) -> ColumnTransformer:
    """Create leakage-safe numeric/categorical preprocessing."""
    X_model = X.drop(columns=list(drop_columns), errors="ignore")
    numeric = X_model.select_dtypes(include=np.number).columns.tolist()
    categorical = X_model.select_dtypes(exclude=np.number).columns.tolist()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ], remainder="drop")


def build_logistic_pipeline(
    X: pd.DataFrame,
    class_weight: str | dict[str, float] | None = None,
    penalty: str = "l2",
    C: float = 1.0,
) -> Pipeline:
    """Build a preprocessing + Logistic Regression pipeline."""
    if penalty not in {"l1", "l2"}:
        raise ValueError("penalty must be either 'l1' or 'l2'.")
    if C <= 0:
        raise ValueError("C must be greater than zero.")
    if penalty == "l1":
        model = LogisticRegression(
            C=C,
            penalty="l1",
            solver="liblinear",
            class_weight=class_weight,
            max_iter=3000,
            random_state=RANDOM_STATE,
        )
    else:
        model = LogisticRegression(
            C=C,
            solver="lbfgs",
            class_weight=class_weight,
            max_iter=3000,
            random_state=RANDOM_STATE,
        )
    return Pipeline([("preprocessor", build_preprocessor(X)), ("model", model)])


def evaluate_binary_model(
    y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict[str, float]:
    """Return business-relevant classification and probability metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def threshold_table(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    """Evaluate precision, recall, F1, and flagged rate across thresholds."""
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        predictions = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": round(float(threshold), 2),
            "precision": precision_score(y_true, predictions, zero_division=0),
            "recall": recall_score(y_true, predictions, zero_division=0),
            "f1": f1_score(y_true, predictions, zero_division=0),
            "flagged_rate": float(predictions.mean()),
        })
    return pd.DataFrame(rows)


def coefficient_table(model_pipeline: Pipeline) -> pd.DataFrame:
    """Extract Logistic Regression coefficients and odds ratios."""
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]
    result = pd.DataFrame({
        "feature": preprocessor.get_feature_names_out(),
        "coefficient": model.coef_.ravel(),
    })
    result["odds_ratio"] = np.exp(result["coefficient"])
    result["abs_coefficient"] = result["coefficient"].abs()
    return result.sort_values(
        "abs_coefficient", ascending=False
    ).reset_index(drop=True)


def risk_segments(probabilities: np.ndarray) -> pd.Categorical:
    """Map churn probabilities to interpretable risk groups."""
    return pd.cut(
        probabilities, bins=[-np.inf, 0.30, 0.60, np.inf],
        labels=["Low", "Medium", "High"], right=False,
    )
