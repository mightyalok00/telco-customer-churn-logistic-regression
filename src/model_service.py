"""Shared Streamlit model service for the Telco Customer Churn app."""

from __future__ import annotations

import streamlit as st
from sklearn.model_selection import GridSearchCV, train_test_split
from threadpoolctl import threadpool_limits

from analysis import (
    DEFAULT_C_VALUES,
    DEFAULT_CV_FOLDS,
    DEFAULT_TEST_SIZE,
    RANDOM_STATE,
    TARGET,
    add_business_features,
    build_logistic_pipeline,
    clean_data,
    load_data,
)


@st.cache_resource(
    show_spinner="Training the tuned Logistic Regression model...",
    max_entries=1,
)
def get_project_model(
    data_path: str,
) -> tuple[
    object,
    object,
    object,
    object,
    object,
    object,
]:
    """Load data and train one shared, memory-bounded project model.

    The function is cached globally so the dashboard and the Q&A page reuse the
    same fitted model instead of maintaining separate GridSearchCV caches.
    """
    data = add_business_features(clean_data(load_data(data_path)))
    X = data.drop(columns=[TARGET])
    y = (data[TARGET] == "Yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=DEFAULT_TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    search = GridSearchCV(
        build_logistic_pipeline(X_train),
        param_grid={"model__C": DEFAULT_C_VALUES},
        scoring="roc_auc",
        cv=DEFAULT_CV_FOLDS,
        n_jobs=1,
        pre_dispatch=1,
        refit=True,
    )
    # Keep native BLAS/OpenMP thread pools single-threaded to reduce peak RAM.
    with threadpool_limits(limits=1):
        search.fit(X_train, y_train)
    return search, data, X_train, X_test, y_train, y_test
