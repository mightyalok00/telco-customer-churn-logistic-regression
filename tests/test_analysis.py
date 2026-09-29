"""Unit and integration tests for Telco churn analysis utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import (
    ID_COLUMN,
    add_business_features,
    build_logistic_pipeline,
    build_preprocessor,
    clean_data,
    coefficient_table,
    evaluate_binary_model,
    load_data,
    risk_segments,
    threshold_table,
)


def sample_data() -> pd.DataFrame:
    """Return a small two-class dataset suitable for pipeline tests."""
    return pd.DataFrame({
        "customerID": [f"C{i:03d}" for i in range(12)],
        "tenure": [1, 3, 6, 9, 12, 18, 24, 30, 36, 48, 60, 72],
        "TotalCharges": [
            50, 120, 300, 450, 650, 900, 1200, 1600,
            2100, 3000, 4000, 5200,
        ],
        "MonthlyCharges": [
            50, 40, 50, 50, 54, 60, 55, 53, 58, 62, 67, 72,
        ],
        "Contract": [
            "Month-to-month", "Month-to-month", "Month-to-month", "Month-to-month",
            "One year", "One year", "One year", "One year",
            "Two year", "Two year", "Two year", "Two year",
        ],
        "PhoneService": ["Yes", "Yes", "Yes", "No"] * 3,
        "OnlineSecurity": ["No", "Yes", "No", "No"] * 3,
        "Churn": ["Yes", "No", "Yes", "No"] * 3,
    })


def test_load_data_validates_target(tmp_path: Path) -> None:
    """A dataset without the target should fail fast."""
    path = tmp_path / "bad.csv"
    pd.DataFrame({"feature": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Expected target"):
        load_data(path)


def test_clean_total_charges_is_numeric_and_duplicates_removed() -> None:
    """Cleaning should coerce charges and remove exact duplicate rows."""
    data = pd.DataFrame({
        "TotalCharges": ["10.5", " ", "20", "20"],
        "Churn": ["No", "Yes", "No", "No"],
    })
    clean = clean_data(data)
    assert pd.api.types.is_numeric_dtype(clean["TotalCharges"])
    assert clean["TotalCharges"].isna().sum() == 1
    assert len(clean) == 3


def test_business_features_are_created() -> None:
    """Business features should be created with expected values."""
    data = pd.DataFrame({
        "PhoneService": ["Yes"], "OnlineSecurity": ["No"],
        "OnlineBackup": ["Yes"], "DeviceProtection": ["No"],
        "TechSupport": ["No"], "StreamingTV": ["No"],
        "StreamingMovies": ["Yes"], "tenure": [12],
        "TotalCharges": [600.0], "MonthlyCharges": [50.0],
    })
    out = add_business_features(data)
    assert out.loc[0, "ServiceCount"] == 3
    assert out.loc[0, "AvgMonthlySpend"] == 50.0
    assert {"TenureBand", "ChargeBand"} <= set(out.columns)


def test_business_features_tolerate_partial_columns() -> None:
    """Feature engineering should work with partial service columns."""
    data = pd.DataFrame({
        "PhoneService": ["Yes", "No"], "tenure": [0, 1],
        "TotalCharges": [0.0, 50.0], "MonthlyCharges": [50.0, 50.0],
    })
    out = add_business_features(data)
    assert out["ServiceCount"].tolist() == [1, 0]
    assert out.loc[0, "AvgMonthlySpend"] == 50.0


def test_preprocessor_excludes_customer_id() -> None:
    """The customer identifier must not become a model feature."""
    preprocessor = build_preprocessor(sample_data().drop(columns=["Churn"]))
    assert all(
        ID_COLUMN not in name
        for name in preprocessor.get_feature_names_out()
    )


def test_logistic_pipeline_fits_and_handles_unknown_categories() -> None:
    """The pipeline should fit and predict on an unseen category."""
    data = sample_data()
    X = data.drop(columns=["Churn"])
    y = (data["Churn"] == "Yes").astype(int)
    model = build_logistic_pipeline(X)
    model.fit(X.iloc[:10], y.iloc[:10])
    X_new = X.iloc[[10]].copy()
    X_new["Contract"] = "New Contract Type"
    probability = model.predict_proba(X_new)[:, 1][0]
    assert 0.0 <= probability <= 1.0


def test_invalid_logistic_configuration_raises() -> None:
    """Invalid penalty and non-positive C should fail clearly."""
    X = sample_data().drop(columns=["Churn"])
    with pytest.raises(ValueError, match="penalty"):
        build_logistic_pipeline(X, penalty="elasticnet")
    with pytest.raises(ValueError, match="greater than zero"):
        build_logistic_pipeline(X, C=0)


def test_binary_metrics_are_computed() -> None:
    """All expected classification and probability metrics are returned."""
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    probabilities = np.array([0.10, 0.40, 0.60, 0.90])
    metrics = evaluate_binary_model(y_true, y_pred, probabilities)
    assert set(metrics) == {
        "accuracy", "precision", "recall", "f1",
        "roc_auc", "pr_auc", "log_loss", "brier_score",
    }
    assert all(np.isfinite(value) for value in metrics.values())


def test_threshold_table_has_expected_columns_and_range() -> None:
    """Threshold analysis should evaluate the 0.10-0.90 range."""
    table = threshold_table(
        pd.Series([0, 0, 1, 1]),
        np.array([0.10, 0.40, 0.60, 0.90]),
    )
    assert {
        "threshold", "precision", "recall", "f1", "flagged_rate"
    } <= set(table.columns)
    assert table["threshold"].min() == 0.10
    assert table["threshold"].max() == 0.90


def test_coefficient_table_contains_odds_ratios() -> None:
    """Fitted models should expose coefficient and odds-ratio tables."""
    data = sample_data()
    X = data.drop(columns=["Churn"])
    y = (data["Churn"] == "Yes").astype(int)
    model = build_logistic_pipeline(X)
    model.fit(X, y)
    table = coefficient_table(model)
    assert {
        "feature", "coefficient", "odds_ratio", "abs_coefficient"
    } <= set(table.columns)
    assert len(table) > 0
    assert np.isfinite(table["odds_ratio"]).all()


def test_risk_segments_cover_all_probabilities() -> None:
    """Every probability receives a risk label."""
    segments = risk_segments(np.array([0.05, 0.30, 0.59, 0.60, 0.95]))
    assert segments.tolist() == ["Low", "Medium", "Medium", "High", "High"]


def test_pipeline_has_scaling_and_one_hot_encoding() -> None:
    """Preprocessing should include scaling and one-hot encoding."""
    preprocessor = build_preprocessor(sample_data().drop(columns=["Churn"]))
    assert "num" in preprocessor.named_transformers
    assert "cat" in preprocessor.named_transformers
    assert (
        preprocessor.named_transformers["num"]
        .named_steps["scaler"].__class__.__name__
        == "StandardScaler"
    )
    assert (
        preprocessor.named_transformers["cat"]
        .named_steps["onehot"].__class__.__name__
        == "OneHotEncoder"
    )
