"""Streamlit dashboard for the Telco Customer Churn project."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis import (  # noqa: E402
    TARGET,
    add_business_features,
    coefficient_table,
    evaluate_binary_model,
    risk_segments,
    threshold_table,
)
from model_service import get_project_model  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

st.set_page_config(
    page_title="Telco Churn Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #334155 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {font-size: 2.35rem; margin-bottom: .35rem;}
    .hero p {font-size: 1.02rem; opacity: .88; margin-bottom: 0;}
    .insight {
        padding: .9rem 1rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 12px;
        margin-bottom: .6rem;
    }
    .small-muted {color: #64748b; font-size: .88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    "### 📚 Project Knowledge Base"
)
st.sidebar.caption(
    "The Streamlit navigation includes the complete 157-question Q&A "
    "covering all 18 analytical sections."
)

st.markdown(
    """
    <div class="hero">
      <h1>📉 Telco Churn Intelligence</h1>
      <p>
        An interactive, interpretable customer-churn analytics workspace powered
        by Logistic Regression.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


def fmt_pct(value: float) -> str:
    """Format a decimal metric as a percentage."""
    return f"{value:.1%}"


def metric_card(label: str, value: str, help_text: str) -> None:
    """Render a compact metric card."""
    st.metric(label, value, help=help_text)


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

model, data, X_test, y_test = get_project_model(str(DATA_PATH))

test_prob = model.predict_proba(X_test)[:, 1]
threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Customers at or above this predicted churn probability are flagged.",
)
test_pred = (test_prob >= threshold).astype(int)
metrics = evaluate_binary_model(y_test, test_pred, test_prob)

st.sidebar.divider()
st.sidebar.subheader("Dashboard controls")
st.sidebar.caption(
    "Filters below affect the customer exploration and risk tables. "
    "Model metrics remain based on the untouched holdout set."
)

tabs = st.tabs(
    ["Executive Overview", "Customer Explorer", "Risk Predictor", "Model Insights"]
)

with tabs[0]:
    st.subheader("Executive Overview")
    total_customers = len(data)
    churn_rate = (data[TARGET] == "Yes").mean()
    high_risk = int((test_prob >= 0.60).sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(
            "Customers", f"{total_customers:,}", "Rows in the supplied dataset."
        )
    with c2:
        metric_card(
            "Observed churn",
            fmt_pct(churn_rate),
            "Observed churn rate in the dataset.",
        )
    with c3:
        metric_card(
            "Holdout ROC-AUC",
            f"{metrics['roc_auc']:.3f}",
            "Discrimination on the untouched holdout set.",
        )
    with c4:
        metric_card(
            "High-risk holdout",
            f"{high_risk:,}",
            "Holdout customers with predicted probability ≥ 60%.",
        )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Churn distribution")
        churn_counts = (
            data[TARGET]
            .value_counts()
            .rename_axis("Churn")
            .reset_index(name="Customers")
        )
        fig = px.bar(
            churn_counts,
            x="Churn",
            y="Customers",
            text="Customers",
            title="Observed customer churn",
        )
        fig.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Contract and churn")
        contract_churn = (
            pd.crosstab(data["Contract"], data[TARGET], normalize="index")
            .reset_index()
        )
        fig = px.bar(
            contract_churn,
            x="Contract",
            y="Yes",
            text_auto=".1%",
            title="Observed churn rate by contract",
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=360, yaxis_title="Churn rate")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Current model quality")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", fmt_pct(metrics["accuracy"]))
    m2.metric("Precision", fmt_pct(metrics["precision"]))
    m3.metric("Recall", fmt_pct(metrics["recall"]))
    m4.metric("F1", fmt_pct(metrics["f1"]))
    m5.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")

    st.info(
        f"Best C from 5-fold ROC-AUC tuning: {model.best_params_['model__C']}. "
        "The dashboard uses the same leakage-safe preprocessing pipeline "
        "as the project."
    )

with tabs[1]:
    st.subheader("Customer Explorer")
    filtered = data.copy()

    st.markdown("#### 🔎 Customer filters")
    st.caption(
        "Combine multiple filters to create a focused customer cohort. "
        "Use Reset filters to return to the full dataset."
    )

    if st.button("↺ Reset customer filters", key="reset_customer_filters"):
        for key in (
            "filter_contract",
            "filter_internet",
            "filter_payment",
            "filter_churn",
            "filter_paperless",
            "filter_senior",
        ):
            st.session_state.pop(key, None)
        st.rerun()

    f1, f2, f3 = st.columns(3)
    with f1:
        contract_options = sorted(filtered["Contract"].dropna().unique())
        selected_contracts = st.multiselect(
            "Contract",
            contract_options,
            default=contract_options,
            key="filter_contract",
        )
        internet_options = sorted(filtered["InternetService"].dropna().unique())
        selected_internet = st.multiselect(
            "Internet service",
            internet_options,
            default=internet_options,
            key="filter_internet",
        )
    with f2:
        payment_options = sorted(filtered["PaymentMethod"].dropna().unique())
        selected_payment = st.multiselect(
            "Payment method",
            payment_options,
            default=payment_options,
            key="filter_payment",
        )
        churn_options = sorted(filtered[TARGET].dropna().unique())
        selected_churn = st.multiselect(
            "Observed churn",
            churn_options,
            default=churn_options,
            key="filter_churn",
        )
    with f3:
        paperless_options = sorted(filtered["PaperlessBilling"].dropna().unique())
        selected_paperless = st.multiselect(
            "Paperless billing",
            paperless_options,
            default=paperless_options,
            key="filter_paperless",
        )
        senior_options = sorted(filtered["SeniorCitizen"].dropna().unique())
        selected_senior = st.multiselect(
            "Senior citizen",
            senior_options,
            default=senior_options,
            key="filter_senior",
            format_func=lambda value: "Yes" if value == 1 else "No",
        )

    f4, f5, f6 = st.columns(3)
    with f4:
        tenure_range = st.slider(
            "Tenure (months)",
            0,
            int(filtered["tenure"].max()),
            (0, int(filtered["tenure"].max())),
            key="filter_tenure",
        )
    with f5:
        monthly_range = st.slider(
            "Monthly charges",
            float(filtered["MonthlyCharges"].min()),
            float(filtered["MonthlyCharges"].max()),
            (
                float(filtered["MonthlyCharges"].min()),
                float(filtered["MonthlyCharges"].max()),
            ),
            step=1.0,
            key="filter_monthly",
        )
    with f6:
        total_range = st.slider(
            "Total charges",
            float(filtered["TotalCharges"].min()),
            float(filtered["TotalCharges"].max()),
            (
                float(filtered["TotalCharges"].min()),
                float(filtered["TotalCharges"].max()),
            ),
            step=10.0,
            key="filter_total",
        )

    filtered = filtered[
        filtered["Contract"].isin(selected_contracts)
        & filtered["InternetService"].isin(selected_internet)
        & filtered["PaymentMethod"].isin(selected_payment)
        & filtered[TARGET].isin(selected_churn)
        & filtered["PaperlessBilling"].isin(selected_paperless)
        & filtered["SeniorCitizen"].isin(selected_senior)
        & filtered["tenure"].between(*tenure_range)
        & filtered["MonthlyCharges"].between(*monthly_range)
        & filtered["TotalCharges"].between(*total_range)
    ]

    e1, e2, e3 = st.columns(3)
    e1.metric("Matching customers", f"{len(filtered):,}")
    e2.metric(
        "Observed churn",
        fmt_pct((filtered[TARGET] == "Yes").mean()) if len(filtered) else "—",
    )
    e3.metric(
        "Average monthly charge",
        f"₹{filtered['MonthlyCharges'].mean():.2f}" if len(filtered) else "—",
    )

    st.markdown("#### Monthly charges vs. tenure")
    if len(filtered):
        fig = px.scatter(
            filtered,
            x="tenure",
            y="MonthlyCharges",
            color=TARGET,
            hover_data=["Contract", "InternetService", "PaymentMethod"],
            opacity=0.65,
        )
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

        display_cols = [
            "customerID", "tenure", "Contract", "InternetService",
            "MonthlyCharges", "TotalCharges", "Churn",
        ]
        st.dataframe(
            filtered[display_cols].sort_values("MonthlyCharges", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No customers match the selected filters.")

with tabs[2]:
    st.subheader("Risk Predictor")
    st.caption(
        "Enter a customer profile to estimate churn probability. "
        "This is a predictive portfolio demonstration, not an automated "
        "decision system."
    )

    with st.form("risk_predictor"):
        p1, p2, p3 = st.columns(3)
        with p1:
            tenure = st.number_input("Tenure (months)", 0, 100, 12)
            monthly = st.number_input(
                "Monthly charges", 0.0, 300.0, 70.0, step=1.0
            )
            total = st.number_input(
                "Total charges",
                0.0,
                50000.0,
                float(tenure * monthly),
                step=10.0,
            )
            contract = st.selectbox(
                "Contract", sorted(data["Contract"].dropna().unique())
            )
        with p2:
            internet = st.selectbox(
                "Internet service", sorted(data["InternetService"].dropna().unique())
            )
            payment = st.selectbox(
                "Payment method", sorted(data["PaymentMethod"].dropna().unique())
            )
            paperless = st.selectbox(
                "Paperless billing", sorted(data["PaperlessBilling"].dropna().unique())
            )
            senior = st.selectbox("Senior citizen", [0, 1])
        with p3:
            partner = st.selectbox(
                "Partner", sorted(data["Partner"].dropna().unique())
            )
            dependents = st.selectbox(
                "Dependents", sorted(data["Dependents"].dropna().unique())
            )
            phone = st.selectbox(
                "Phone service", sorted(data["PhoneService"].dropna().unique())
            )
            multiple = st.selectbox(
                "Multiple lines", sorted(data["MultipleLines"].dropna().unique())
            )
        submitted = st.form_submit_button("Predict churn risk", type="primary")

    if submitted:
        # Start from a real one-row template so required columns remain present.
        profile = data.drop(columns=[TARGET]).iloc[[0]].copy()
        profile["tenure"] = tenure
        profile["MonthlyCharges"] = monthly
        profile["TotalCharges"] = total
        profile["Contract"] = contract
        profile["InternetService"] = internet
        profile["PaymentMethod"] = payment
        profile["PaperlessBilling"] = paperless
        profile["SeniorCitizen"] = senior
        profile["Partner"] = partner
        profile["Dependents"] = dependents
        profile["PhoneService"] = phone
        profile["MultipleLines"] = multiple

        profile = add_business_features(profile)
        probability = float(model.predict_proba(profile)[:, 1][0])
        segment = str(risk_segments(np.array([probability]))[0])

        a, b, c = st.columns(3)
        a.metric("Predicted churn probability", f"{probability:.1%}")
        b.metric("Risk segment", segment)
        c.metric(
            "Decision at current threshold",
            "Flag" if probability >= threshold else "Do not flag",
        )

        st.progress(probability)
        if segment == "High":
            st.error("High predicted risk: probability is at or above 60%.")
        elif segment == "Medium":
            st.warning(
                "Medium predicted risk: probability is between 30% and 60%."
            )
        else:
            st.success("Low predicted risk: probability is below 30%.")

with tabs[3]:
    st.subheader("Model Insights")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Threshold analysis")
        threshold_df = threshold_table(y_test, test_prob)
        threshold_df["flagged_pct"] = threshold_df["flagged_rate"] * 100
        fig = px.line(
            threshold_df,
            x="threshold",
            y=["precision", "recall", "f1"],
            markers=True,
            title="Precision / Recall / F1 by threshold",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Confusion matrix")
        matrix = confusion_matrix(y_test, test_pred)
        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual No", "Actual Yes"],
            columns=["Predicted No", "Predicted Yes"],
        )
        fig = px.imshow(
            matrix_df,
            text_auto=True,
            aspect="auto",
            title=f"Holdout confusion matrix @ {threshold:.2f}",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Most influential Logistic Regression features")
    coefficients = coefficient_table(model.best_estimator_).head(20).copy()
    coefficients["direction"] = np.where(
        coefficients["coefficient"] >= 0, "Higher churn odds", "Lower churn odds"
    )
    fig = px.bar(
        coefficients.sort_values("coefficient"),
        x="coefficient",
        y="feature",
        color="direction",
        orientation="h",
        title="Top 20 coefficients by absolute magnitude",
    )
    fig.update_layout(height=620)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Coefficient / odds-ratio table")
    st.dataframe(
        coefficients[["feature", "coefficient", "odds_ratio", "direction"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Holdout risk segmentation")
    segment_series = risk_segments(test_prob)
    segment_counts = (
        pd.Series(segment_series, name="Risk")
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .rename_axis("Risk")
        .reset_index(name="Customers")
    )
    fig = px.bar(
        segment_counts,
        x="Risk",
        y="Customers",
        text="Customers",
        title="Predicted customer-risk distribution",
    )
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)

    csv = coefficients.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download coefficient analysis CSV",
        data=csv,
        file_name="telco_churn_coefficients.csv",
        mime="text/csv",
    )

st.divider()
st.markdown(
    '<p class="small-muted">Telco Customer Churn Intelligence • '
    'Logistic Regression • Educational / portfolio use</p>',
    unsafe_allow_html=True,
)
