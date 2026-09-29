"""Train and evaluate the Telco churn Logistic Regression model."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis import (  # noqa: E402
    DEFAULT_C_VALUES,
    DEFAULT_CV_FOLDS,
    DEFAULT_TEST_SIZE,
    RANDOM_STATE,
    TARGET,
    add_business_features,
    build_logistic_pipeline,
    clean_data,
    evaluate_binary_model,
    load_data,
)

DATA = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def prepare_dataset():
    """Load, clean, engineer features, and create the binary target."""
    data = add_business_features(clean_data(load_data(DATA)))
    return data.drop(columns=[TARGET]), (data[TARGET] == "Yes").astype(int)


def train_model(X_train, y_train) -> GridSearchCV:
    """Fit Logistic Regression with five-fold ROC-AUC tuning."""
    search = GridSearchCV(
        build_logistic_pipeline(X_train),
        param_grid={"model__C": DEFAULT_C_VALUES},
        scoring="roc_auc",
        cv=DEFAULT_CV_FOLDS,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_model(search, X_test, y_test) -> dict[str, float]:
    """Evaluate the tuned estimator on the untouched holdout set."""
    probabilities = search.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)
    return evaluate_binary_model(y_test, predictions, probabilities)


def main() -> None:
    """Run the complete training workflow and save the fitted pipeline."""
    LOGGER.info("Loading and preparing dataset")
    X, y = prepare_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=DEFAULT_TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    LOGGER.info("Training Logistic Regression with cross-validation")
    search = train_model(X_train, y_train)
    metrics = evaluate_model(search, X_test, y_test)
    LOGGER.info("Best parameters: %s", search.best_params_)
    LOGGER.info(
        "Cross-validated ROC-AUC: %.4f",
        search.best_score_,
    )
    LOGGER.info(
        "Holdout metrics: %s",
        {key: round(value, 4) for key, value in metrics.items()},
    )
    LOGGER.info(
        "Classification report:\n%s",
        classification_report(y_test, search.predict(X_test)),
    )
    MODEL_DIR.mkdir(exist_ok=True)
    output_path = MODEL_DIR / "logistic_regression_pipeline.joblib"
    joblib.dump(search.best_estimator_, output_path)
    LOGGER.info("Saved model to %s", output_path)


if __name__ == "__main__":
    main()
