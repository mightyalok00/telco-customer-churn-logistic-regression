
"""Reusable Telco Customer Churn analysis utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
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


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV and validate the target column."""
    data = pd.read_csv(path)
    if TARGET not in data.columns:
        raise ValueError(f"Expected target column {TARGET!r}.")
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean blank TotalCharges values and coerce it to numeric."""
    clean = data.copy()
    clean["TotalCharges"] = pd.to_numeric(clean["TotalCharges"], errors="coerce")
    clean = clean.drop_duplicates().reset_index(drop=True)
    return clean


def add_business_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable customer-level features."""
    out = data.copy()
    service_cols = [
        "PhoneService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    out["ServiceCount"] = sum(
        (out[col].isin(["Yes", "Yes - no internet service"])).astype(int)
        for col in service_cols
        if col in out
    )
    out["AvgMonthlySpend"] = np.where(
        out["tenure"] > 0, out["TotalCharges"] / out["tenure"], out["MonthlyCharges"]
    )
    out["TenureBand"] = pd.cut(
        out["tenure"],
        bins=[-1, 6, 12, 24, 48, np.inf],
        labels=["0-6", "7-12", "13-24", "25-48", "49+"],
    )
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

    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),
               ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")),
               ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def build_logistic_pipeline(
    X: pd.DataFrame,
    class_weight=None,
    penalty: str = "l2",
    C: float = 1.0,
) -> Pipeline:
    """Build a preprocessing + Logistic Regression pipeline."""
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    preprocessor = build_preprocessor(X)
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        class_weight=class_weight,
        max_iter=3000,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate_binary_model(
    y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict:
    """Return business-relevant classification and probability metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
    }


def threshold_table(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    """Evaluate multiple classification thresholds."""
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        pred = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
                "flagged_rate": pred.mean(),
            }
        )
    return pd.DataFrame(rows)


def coefficient_table(model_pipeline: Pipeline) -> pd.DataFrame:
    """Extract Logistic Regression coefficients with feature names."""
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    coefficients = model.coef_.ravel()
    result = pd.DataFrame(
        {"feature": names, "coefficient": coefficients}
    )
    result["odds_ratio"] = np.exp(result["coefficient"])
    result["abs_coefficient"] = result["coefficient"].abs()
    return result.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def risk_segments(probabilities: np.ndarray) -> pd.Categorical:
    """Map probabilities to simple, interpretable risk groups."""
    return pd.cut(
        probabilities,
        bins=[-np.inf, 0.30, 0.60, np.inf],
        labels=["Low", "Medium", "High"],
        right=False,
    )
