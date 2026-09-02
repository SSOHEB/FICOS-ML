"""Evaluation and Exploratory Data Analysis (EDA) module for FICOS ML."""

from src.evaluation.eda import (
    profile_dataset,
    compute_target_statistics,
    compute_correlation_matrix,
    analyze_missingness,
    detect_outliers,
    compute_volatility_metrics,
    test_stationarity,
    compute_autocorrelation,
    generate_eda_plots,
)

__all__ = [
    "profile_dataset",
    "compute_target_statistics",
    "compute_correlation_matrix",
    "analyze_missingness",
    "detect_outliers",
    "compute_volatility_metrics",
    "test_stationarity",
    "compute_autocorrelation",
    "generate_eda_plots",
]
