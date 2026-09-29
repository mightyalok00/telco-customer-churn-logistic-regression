"""Complete project question-and-answer knowledge base."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.model_selection import GridSearchCV, train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis import (  # noqa: E402
    DEFAULT_C_VALUES,
    DEFAULT_CV_FOLDS,
    DEFAULT_TEST_SIZE,
    RANDOM_STATE,
    TARGET,
    add_business_features,
    build_logistic_pipeline,
    clean_data,
    coefficient_table,
    evaluate_binary_model,
    load_data,
    risk_segments,
    threshold_table,
)

DATA_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
QUESTIONS_PATH = ROOT / "docs" / "PROJECT_QUESTIONS.md"

st.set_page_config(
    page_title="Telco Churn • 157 Q&A",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1450px; padding-top: 1.3rem;}
    .hero {
        padding: 1.8rem 2rem; border-radius: 20px; color: white;
        background: linear-gradient(135deg,#0f172a,#172554,#164e63);
        margin-bottom: 1rem;
    }
    .hero h1 {margin:0 0 .35rem; font-size:2.35rem;}
    .hero p {margin:0; opacity:.88;}
    .card {
        border:1px solid rgba(148,163,184,.2); border-radius:14px;
        padding:1rem; background:rgba(148,163,184,.05); margin:.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>📚 Telco Churn — Complete Project Q&A</h1>
      <p>157 questions across 18 analytical sections, answered from the supplied dataset,
      the fitted Logistic Regression workflow, and the documented project
      methodology.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_project_data(path: str) -> pd.DataFrame:
    return add_business_features(clean_data(load_data(path)))


@st.cache_data(show_spinner=False)
def load_questions(path: str) -> list[dict[str, str]]:
    text = Path(path).read_text(encoding="utf-8")
    rows = []
    section = ""
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        elif section and re.match(r"^\d+\.\s+", line.strip()):
            question = re.sub(r"^\d+\.\s+", "", line.strip())
            rows.append({"section": section, "question": question})
    return rows


@st.cache_resource(show_spinner="Training the project model for the Q&A evidence...")
def train_model(data: pd.DataFrame):
    X = data.drop(columns=[TARGET])
    y = data[TARGET].eq("Yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=DEFAULT_TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    search = GridSearchCV(
        build_logistic_pipeline(X_train),
        {"model__C": DEFAULT_C_VALUES},
        scoring="roc_auc",
        cv=DEFAULT_CV_FOLDS,
        n_jobs=1,
        pre_dispatch=1,
    )
    search.fit(X_train, y_train)
    return search, X_test, y_test


def answer(question: str, data: pd.DataFrame, model, y_test, probabilities):
    """Return a concise evidence-based answer and its evidence class."""
    q = question.lower()
    churn = data[TARGET].eq("Yes")
    metrics = evaluate_binary_model(
        y_test, (probabilities >= 0.50).astype(int), probabilities
    )

    if "overall customer churn rate" in q:
        return "Data-backed", (
            f"{int(churn.sum()):,} of {len(data):,} customers churned, "
            f"so the observed churn rate is {churn.mean():.2%}."
        )

    if "business problem" in q:
        return "Methodology", (
            "The project analyzes customer churn as a binary predictive problem: "
            "understand observable churn patterns, estimate churn probability, "
            "and translate predictions into interpretable risk segments."
        )

    if "target variable" in q or "target class" in q:
        return "Data-backed", (
            "Churn is the binary target: Yes represents observed churn and "
            "No represents observed retention."
        )

    if "structure, size" in q or "dimensionality" in q:
        return "Data-backed", (
            f"The supplied dataset contains {len(data):,} cleaned rows and "
            "21 original columns."
        )

    if "duplicate" in q:
        return "Data-backed", (
            "The cleaning function removes duplicate rows and resets the index "
            "before feature engineering and modeling."
        )

    if "missing" in q or "blank" in q:
        missing = data.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            return (
                "Data-backed",
                "No missing values remain in the cleaned dashboard dataset.",
            )
        details = ", ".join(
            f"{name}: {int(value)}" for name, value in missing.items()
        )
        return "Data-backed", (
            "TotalCharges is coerced to numeric so blank values become missing "
            f"values for pipeline imputation. Current missing values: {details}."
        )

    if "imbalanced" in q or "class imbalance" in q:
        return "Data-backed", (
            f"Yes. Churn=Yes is {churn.mean():.2%} and Churn=No is "
            f"{(~churn).mean():.2%}. The project therefore evaluates multiple "
            "classification and probability metrics instead of Accuracy alone."
        )

    if "distribution of customer tenure" in q:
        return "Data-backed", (
            f"Tenure ranges from {data.tenure.min():.0f} to "
            f"{data.tenure.max():.0f} months; median tenure is "
            f"{data.tenure.median():.1f} months."
        )

    if "distribution of monthly charges" in q:
        return "Data-backed", (
            f"MonthlyCharges range from {data.MonthlyCharges.min():.2f} to "
            f"{data.MonthlyCharges.max():.2f}; median is "
            f"{data.MonthlyCharges.median():.2f}."
        )

    if "distribution of total charges" in q:
        return "Data-backed", (
            f"TotalCharges range from {data.TotalCharges.min():.2f} to "
            f"{data.TotalCharges.max():.2f}; median is "
            f"{data.TotalCharges.median():.2f}."
        )

    if "contract" in q and ("churn" in q or "differ" in q or "vary" in q):
        rates = data.groupby("Contract")[TARGET].apply(lambda s: s.eq("Yes").mean())
        return "Data-backed", (
            "Observed churn by contract: "
            + "; ".join(f"{name}: {rate:.1%}" for name, rate in rates.items())
            + "."
        )

    if "payment method" in q:
        rates = data.groupby("PaymentMethod")[TARGET].apply(
            lambda s: s.eq("Yes").mean()
        )
        return "Data-backed", (
            "Observed churn by payment method: "
            + "; ".join(f"{name}: {rate:.1%}" for name, rate in rates.items())
            + "."
        )

    if "internet-service" in q or "internet service" in q:
        rates = data.groupby("InternetService")[TARGET].apply(
            lambda s: s.eq("Yes").mean()
        )
        return "Data-backed", (
            "Observed churn by internet service: "
            + "; ".join(f"{name}: {rate:.1%}" for name, rate in rates.items())
            + "."
        )

    if "paperless billing" in q:
        rates = data.groupby("PaperlessBilling")[TARGET].apply(
            lambda s: s.eq("Yes").mean()
        )
        return "Data-backed", (
            "Observed churn by paperless billing: "
            + "; ".join(f"{name}: {rate:.1%}" for name, rate in rates.items())
            + "."
        )

    if "logistic regression" in q and (
        "accurately" in q or "accuracy" in q or "predict" in q
    ):
        return "Model-backed", (
            f"At threshold 0.50, holdout Accuracy={metrics['accuracy']:.4f}, "
            f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, "
            f"and PR-AUC={metrics['pr_auc']:.4f}."
        )

    if "roc-auc" in q or "pr-auc" in q:
        return "Model-backed", (
            f"Holdout ROC-AUC={metrics['roc_auc']:.4f}; "
            f"holdout PR-AUC={metrics['pr_auc']:.4f}."
        )

    if "log loss" in q:
        return "Model-backed", f"Holdout Log Loss={metrics['log_loss']:.4f}."

    if "brier" in q:
        return "Model-backed", f"Holdout Brier score={metrics['brier_score']:.4f}."

    if "regularization strength" in q or "hyperparameter" in q:
        return "Model-backed", (
            "The project tunes C over "
            f"{DEFAULT_C_VALUES} using {DEFAULT_CV_FOLDS}-fold ROC-AUC CV. "
            f"Selected C={model.best_params_['model__C']}."
        )

    if "cross-validation" in q or "cross validation" in q:
        return "Model-backed", (
            f"The training workflow uses {DEFAULT_CV_FOLDS}-fold CV for "
            f"ROC-AUC selection; best mean CV ROC-AUC={model.best_score_:.4f}."
        )

    if "preprocessing" in q or "pipeline" in q or "one-hot" in q:
        return "Methodology", (
            "Numeric features use median imputation and StandardScaler. "
            "Categorical features use most-frequent imputation and one-hot encoding. "
            "All preprocessing is inside the modeling pipeline."
        )

    if "feature engineering" in q or "engineered features" in q:
        return "Methodology", (
            "The project engineers ServiceCount, AvgMonthlySpend, TenureBand, "
            "and ChargeBand for interpretable business analysis."
        )

    if "threshold" in q and ("precision" in q or "recall" in q or "f1" in q):
        table = threshold_table(y_test, probabilities)
        best = table.loc[table["f1"].idxmax()]
        return "Model-backed", (
            f"Threshold selection changes the Precision/Recall trade-off. "
            f"The best F1 in the evaluated grid occurs at threshold "
            f"{best['threshold']:.2f}, with Precision={best['precision']:.3f} "
            f"and Recall={best['recall']:.3f}."
        )

    if "coefficient" in q or "odds ratio" in q:
        coefficients = coefficient_table(model.best_estimator_).head(8)
        details = "; ".join(
            f"{row.feature}: coef={row.coefficient:.3f}, OR={row.odds_ratio:.3f}"
            for row in coefficients.itertuples()
        )
        return "Model-backed", (
            "The fitted Logistic Regression provides coefficients and odds ratios. "
            f"Leading absolute coefficients include: {details}."
        )

    if "risk" in q and ("low" in q or "medium" in q or "high" in q):
        counts = pd.Series(risk_segments(probabilities)).value_counts().reindex(
            ["Low", "Medium", "High"], fill_value=0
        )
        return "Model-backed", (
            "Risk definitions are Low <30%, Medium 30% to <60%, and High >=60%. "
            "Holdout counts: "
            + ", ".join(f"{name}={value:,}" for name, value in counts.items())
            + "."
        )

    if "causal" in q or "causality" in q:
        return "Methodology", (
            "No. The dataset is observational. Associations, coefficients and "
            "predicted probabilities should not be interpreted as causal effects."
        )

    if "limitation" in q:
        return "Methodology", (
            "Key limitations are observational data, no causal identification, "
            "no external validation, and no validated intervention-cost or "
            "fairness framework."
        )

    if "production" in q or "extend" in q:
        return "Extension", (
            "A production extension would add versioned artifacts, scheduled scoring, "
            "data/model drift monitoring, calibration monitoring, fairness checks, "
            "cost-sensitive thresholding, intervention tracking and retraining "
            "governance."
        )

    if "retention" in q or "business impact" in q or "decision" in q:
        return "Methodology", (
            "The model can support analytical prioritization by predicted probability "
            "and risk segment, but the current project does not prescribe a specific "
            "retention action. Business cost, customer value and intervention outcomes "
            "would be required."
        )

    return "Scope note", (
        "This question is part of the authoritative 157-question bank, but the "
        "current dashboard does not compute a dedicated statistic for it. The app "
        "does not invent unsupported evidence; this question can be addressed by "
        "extending the notebook/project analysis."
    )


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()
if not QUESTIONS_PATH.exists():
    st.error(f"Question bank not found: {QUESTIONS_PATH}")
    st.stop()

data = load_project_data(str(DATA_PATH))
questions = load_questions(str(QUESTIONS_PATH))
model, X_test, y_test = train_model(data)
probabilities = model.predict_proba(X_test)[:, 1]

st.sidebar.metric("Questions", f"{len(questions)}/157")
st.sidebar.metric("Sections", "18")
st.sidebar.metric("CV ROC-AUC", f"{model.best_score_:.3f}")
st.sidebar.caption(
    "Answers are generated from the repository's current dataset and model. "
    "Scope notes identify questions that need additional analysis."
)

search = st.text_input(
    "🔎 Search the full question bank",
    placeholder=(
        "Search: churn rate, contract, Logistic Regression, "
        "calibration, business impact..."
    ),
)
sections = list(dict.fromkeys(item["section"] for item in questions))
section = st.selectbox("Filter by section", ["All 18 sections"] + sections)

filtered = questions
if search:
    term = search.lower()
    filtered = [
        item for item in filtered
        if term in item["question"].lower() or term in item["section"].lower()
    ]
if section != "All 18 sections":
    filtered = [item for item in filtered if item["section"] == section]

st.write(f"Showing **{len(filtered)}** of **{len(questions)}** questions.")

for number, item in enumerate(filtered, start=1):
    status, response = answer(item["question"], data, model, y_test, probabilities)
    with st.expander(f"{number}. {item['question']}"):
        st.caption(item["section"])
        st.markdown(f"**Answer status:** '{status}'")
        st.markdown(
            (
                f'<div class="card"><strong>Evidence-based answer</strong>'
                f'<br>{response}</div>'
            ),
            unsafe_allow_html=True,
        )

st.divider()
st.caption(
    "Telco Customer Churn • Complete project Q&A • "
    "Predictive association is not causal evidence."
)
