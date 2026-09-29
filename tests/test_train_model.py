"""Integration tests for the Telco churn training workflow."""

from pathlib import Path

import train_model as trainer


def make_training_csv(path: Path) -> None:
    """Write a compact two-class dataset for fast training tests."""
    rows = [
        "customerID,tenure,TotalCharges,MonthlyCharges,Contract,"
        "PhoneService,OnlineSecurity,Churn"
    ]
    contracts = ["Month-to-month", "One year", "Two year"]
    for index in range(20):
        tenure = index + 1
        monthly = 40.0 + index
        total = tenure * monthly
        contract = contracts[index % len(contracts)]
        phone = "Yes" if index % 3 else "No"
        security = "Yes" if index % 2 else "No"
        churn = "Yes" if index % 2 else "No"
        rows.append(
            f"C{index:03d},{tenure},{total},{monthly},{contract},"
            f"{phone},{security},{churn}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_prepare_dataset_creates_binary_target(tmp_path: Path, monkeypatch) -> None:
    """Dataset preparation should return features and a binary target."""
    csv_path = tmp_path / "train.csv"
    make_training_csv(csv_path)
    monkeypatch.setattr(trainer, "DATA", csv_path)

    X, y = trainer.prepare_dataset()

    assert len(X) == 20
    assert set(y.unique()) == {0, 1}
    assert trainer.TARGET not in X.columns


def test_train_and_evaluate_helpers(tmp_path: Path, monkeypatch) -> None:
    """Training and evaluation helpers should produce valid metrics."""
    csv_path = tmp_path / "train.csv"
    make_training_csv(csv_path)
    monkeypatch.setattr(trainer, "DATA", csv_path)

    X, y = trainer.prepare_dataset()
    X_train, X_test, y_train, y_test = trainer.train_test_split(
        X, y, test_size=trainer.DEFAULT_TEST_SIZE,
        stratify=y, random_state=trainer.RANDOM_STATE
    )
    search = trainer.train_model(X_train, y_train)
    metrics = trainer.evaluate_model(search, X_test, y_test)

    assert search.best_params_["model__C"] in trainer.DEFAULT_C_VALUES
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_main_saves_model(tmp_path: Path, monkeypatch) -> None:
    """The command-line workflow should save a reusable model artifact."""
    csv_path = tmp_path / "train.csv"
    model_dir = tmp_path / "models"
    make_training_csv(csv_path)
    monkeypatch.setattr(trainer, "DATA", csv_path)
    monkeypatch.setattr(trainer, "MODEL_DIR", model_dir)

    trainer.main()

    assert (model_dir / "logistic_regression_pipeline.joblib").exists()
