
import numpy as np
import pandas as pd

from analysis import clean_data, add_business_features, threshold_table


def test_clean_total_charges_is_numeric():
    data = pd.DataFrame(
        {"TotalCharges": ["10.5", " ", "20"], "Churn": ["No", "Yes", "No"]}
    )
    clean = clean_data(data)
    assert pd.api.types.is_numeric_dtype(clean["TotalCharges"])


def test_business_features_are_created():
    data = pd.DataFrame(
        {
            "PhoneService": ["Yes"],
            "OnlineSecurity": ["No"],
            "OnlineBackup": ["Yes"],
            "DeviceProtection": ["No"],
            "TechSupport": ["No"],
            "StreamingTV": ["No"],
            "StreamingMovies": ["Yes"],
            "tenure": [12],
            "TotalCharges": [600.0],
            "MonthlyCharges": [50.0],
        }
    )
    out = add_business_features(data)
    assert "ServiceCount" in out.columns
    assert "AvgMonthlySpend" in out.columns
    assert out.loc[0, "ServiceCount"] == 3


def test_threshold_table_has_expected_columns():
    y = pd.Series([0, 0, 1, 1])
    p = np.array([0.10, 0.40, 0.60, 0.90])
    table = threshold_table(y, p)
    assert {"threshold", "precision", "recall", "f1", "flagged_rate"} <= set(table.columns)
