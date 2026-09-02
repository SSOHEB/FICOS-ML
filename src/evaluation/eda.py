"""Exploratory Data Analysis (EDA) and Statistical Validation for FICOS ML."""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf

from src.data.schemas import (
    DATE_COLUMN,
    TARGET_COLUMNS,
    MARKET_COLUMNS,
    GPR_COLUMNS,
    WEATHER_COLUMNS,
    FLAG_COLUMNS,
)


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a comprehensive basic profile of the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Dict[str, Any]: Basic profiling metrics.
    """
    total_rows, total_cols = df.shape
    date_col = DATE_COLUMN if DATE_COLUMN in df.columns else None

    date_min = None
    date_max = None
    total_calendar_days = 0
    duplicate_dates = 0

    if date_col:
        dates = pd.to_datetime(df[date_col])
        date_min = str(dates.min().date())
        date_max = str(dates.max().date())
        total_calendar_days = (dates.max() - dates.min()).days + 1
        duplicate_dates = int(dates.duplicated().sum())

    trading_days = int(df["is_bdi_trading_day"].sum()) if "is_bdi_trading_day" in df.columns else 0
    market_trading_days = int(df["is_market_trading_day"].sum()) if "is_market_trading_day" in df.columns else 0
    non_trading_days = total_rows - trading_days

    missing_counts = df.isnull().sum().to_dict()
    missing_pcts = ((df.isnull().sum() / total_rows) * 100).round(2).to_dict()

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "date_min": date_min,
        "date_max": date_max,
        "total_calendar_days": total_calendar_days,
        "duplicate_dates": duplicate_dates,
        "trading_days": trading_days,
        "non_trading_days": non_trading_days,
        "market_trading_days": market_trading_days,
        "missing_counts": missing_counts,
        "missing_pcts": missing_pcts,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def compute_target_statistics(
    df: pd.DataFrame, target_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Calculate key descriptive statistics for forecasting targets.

    Args:
        df: Input DataFrame.
        target_cols: List of target columns to evaluate. Defaults to TARGET_COLUMNS.

    Returns:
        pd.DataFrame: Tabulated descriptive statistics.
    """
    cols = target_cols or [c for c in TARGET_COLUMNS if c in df.columns]
    stats_list = []

    for col in cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue

        mean_val = s.mean()
        std_val = s.std()
        cv = (std_val / mean_val) * 100 if mean_val != 0 else np.nan

        stats_list.append({
            "target": col,
            "count": int(s.count()),
            "mean": round(float(mean_val), 2),
            "median": round(float(s.median()), 2),
            "std": round(float(std_val), 2),
            "cv_pct": round(float(cv), 2),
            "min": round(float(s.min()), 2),
            "p25": round(float(s.quantile(0.25)), 2),
            "p50": round(float(s.quantile(0.50)), 2),
            "p75": round(float(s.quantile(0.75)), 2),
            "p90": round(float(s.quantile(0.90)), 2),
            "p95": round(float(s.quantile(0.95)), 2),
            "max": round(float(s.max()), 2),
            "skewness": round(float(s.skew()), 2),
            "kurtosis": round(float(s.kurtosis()), 2),
        })

    return pd.DataFrame(stats_list)


def compute_correlation_matrix(
    df: pd.DataFrame, columns: Optional[List[str]] = None, method: str = "pearson"
) -> pd.DataFrame:
    """Compute pairwise correlation matrix across numerical variables.

    Args:
        df: Input DataFrame.
        columns: List of columns to include. If None, uses all numeric columns.
        method: Correlation method ('pearson', 'spearman', 'kendall').

    Returns:
        pd.DataFrame: Correlation matrix.
    """
    if columns is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        numeric_df = df[valid_cols]

    return numeric_df.corr(method=method)


def analyze_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze missing values and their alignment with calendar trading days.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Missingness diagnostics table.
    """
    total_rows = len(df)
    results = []

    has_trading_flag = "is_bdi_trading_day" in df.columns

    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / total_rows) * 100, 2)

        nulls_on_trading_days = 0
        nulls_on_non_trading = 0

        if has_trading_flag and null_count > 0:
            nulls_on_trading_days = int(df[df["is_bdi_trading_day"]][col].isnull().sum())
            nulls_on_non_trading = int(df[~df["is_bdi_trading_day"]][col].isnull().sum())

        results.append({
            "column": col,
            "null_count": null_count,
            "null_pct": null_pct,
            "nulls_trading_days": nulls_on_trading_days,
            "nulls_non_trading_days": nulls_on_non_trading,
            "has_trading_anomalies": bool(nulls_on_trading_days > 0 and col in TARGET_COLUMNS),
        })

    return pd.DataFrame(results)


def detect_outliers(
    series: pd.Series, method: str = "iqr", threshold: float = 1.5
) -> Dict[str, Any]:
    """Detect statistical outliers in a time series.

    Args:
        series: Numerical series to inspect.
        method: Detection method ('iqr' or 'zscore').
        threshold: Multiplier threshold (1.5 for IQR, 3.0 for zscore).

    Returns:
        Dict[str, Any]: Outlier indices, count, bounds, and values.
    """
    clean_s = series.dropna()
    if len(clean_s) == 0:
        return {"outlier_count": 0, "outlier_pct": 0.0, "indices": [], "bounds": ()}

    if method == "iqr":
        q25 = clean_s.quantile(0.25)
        q75 = clean_s.quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - threshold * iqr
        upper_bound = q75 + threshold * iqr
        outlier_mask = (clean_s < lower_bound) | (clean_s > upper_bound)
    elif method == "zscore":
        mean = clean_s.mean()
        std = clean_s.std()
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        outlier_mask = (clean_s < lower_bound) | (clean_s > upper_bound)
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    outlier_series = clean_s[outlier_mask]
    return {
        "method": method,
        "threshold": threshold,
        "lower_bound": round(float(lower_bound), 2),
        "upper_bound": round(float(upper_bound), 2),
        "outlier_count": int(len(outlier_series)),
        "outlier_pct": round(float((len(outlier_series) / len(clean_s)) * 100), 2),
        "indices": list(outlier_series.index),
        "min_outlier": round(float(outlier_series.min()), 2) if len(outlier_series) > 0 else None,
        "max_outlier": round(float(outlier_series.max()), 2) if len(outlier_series) > 0 else None,
    }


def compute_volatility_metrics(series: pd.Series) -> Dict[str, Any]:
    """Compute day-to-day changes and empirical volatility metrics for trading days.

    Args:
        series: Target time series.

    Returns:
        Dict[str, Any]: Summary of return volatility and extreme daily shifts.
    """
    clean_s = series.dropna()
    if len(clean_s) < 2:
        return {"error": "Insufficient observations"}

    daily_pct_change = clean_s.pct_change().dropna() * 100
    daily_abs_change = clean_s.diff().dropna().abs()

    return {
        "n_obs": int(len(daily_pct_change)),
        "pct_change_mean": round(float(daily_pct_change.mean()), 3),
        "pct_change_std": round(float(daily_pct_change.std()), 3),
        "pct_change_min": round(float(daily_pct_change.min()), 2),
        "pct_change_max": round(float(daily_pct_change.max()), 2),
        "pct_change_p01": round(float(daily_pct_change.quantile(0.01)), 2),
        "pct_change_p99": round(float(daily_pct_change.quantile(0.99)), 2),
        "abs_change_mean": round(float(daily_abs_change.mean()), 2),
        "abs_change_median": round(float(daily_abs_change.median()), 2),
        "abs_change_max": round(float(daily_abs_change.max()), 2),
        "annualized_volatility_pct": round(float(daily_pct_change.std() * np.sqrt(252)), 2),
    }


def evaluate_stationarity(series: pd.Series, test_type: str = "adf") -> Dict[str, Any]:
    """Perform Augmented Dickey-Fuller (ADF) or KPSS stationarity tests.

    Args:
        series: Non-empty numeric series.
        test_type: 'adf' for Augmented Dickey-Fuller, 'kpss' for Kwiatkowski-Phillips-Schmidt-Shin.

    Returns:
        Dict[str, Any]: Test statistic, p-value, critical values, and interpretation.
    """
    clean_s = series.dropna()
    if len(clean_s) < 20:
        return {"error": "Series length too short for reliable testing"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if test_type.lower() == "adf":
            res = adfuller(clean_s, autolag="AIC")
            stat, pval, usedlag, nobs, crit_vals, _ = res
            is_stationary = bool(pval < 0.05)
            interpretation = "Stationary (Reject H0 unit root at 5%)" if is_stationary else "Non-Stationary (Fail to reject H0 unit root)"
            return {
                "test": "Augmented Dickey-Fuller (ADF)",
                "statistic": round(float(stat), 4),
                "p_value": round(float(pval), 4),
                "lags_used": int(usedlag),
                "n_obs": int(nobs),
                "critical_values": {k: round(float(v), 4) for k, v in crit_vals.items()},
                "is_stationary_5pct": is_stationary,
                "interpretation": interpretation,
            }
        elif test_type.lower() == "kpss":
            stat, pval, lags, crit_vals = kpss(clean_s, regression="c", nlags="auto")
            is_stationary = bool(pval >= 0.05)
            interpretation = "Stationary (Fail to reject H0 stationarity at 5%)" if is_stationary else "Non-Stationary (Reject H0 stationarity at 5%)"
            return {
                "test": "KPSS Test",
                "statistic": round(float(stat), 4),
                "p_value": round(float(pval), 4),
                "lags_used": int(lags),
                "critical_values": {k: round(float(v), 4) for k, v in crit_vals.items()},
                "is_stationary_5pct": is_stationary,
                "interpretation": interpretation,
            }
        else:
            raise ValueError(f"Unsupported test_type: {test_type}. Use 'adf' or 'kpss'.")


# Alias for backward compatibility
test_stationarity = evaluate_stationarity


def compute_autocorrelation(series: pd.Series, nlags: int = 30) -> Dict[str, np.ndarray]:
    """Compute Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).

    Args:
        series: Numeric series.
        nlags: Number of lags to compute.

    Returns:
        Dict[str, np.ndarray]: ACF and PACF values.
    """
    clean_s = series.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acf_vals = acf(clean_s, nlags=nlags, fft=True)
        pacf_vals = pacf(clean_s, nlags=nlags, method="ywm")

    return {
        "lags": np.arange(len(acf_vals)),
        "acf": np.round(acf_vals, 4),
        "pacf": np.round(pacf_vals, 4),
    }


def generate_eda_plots(
    df: pd.DataFrame, output_dir: Union[str, Path] = "reports/figures"
) -> Dict[str, str]:
    """Generate and save essential publication-quality EDA figures.

    Args:
        df: Input master dataset.
        output_dir: Directory where PNG plots will be saved.

    Returns:
        Dict[str, str]: Map of plot names to saved file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_plots: Dict[str, str] = {}

    sns.set_theme(style="whitegrid", font_scale=1.0)
    dates = pd.to_datetime(df[DATE_COLUMN])

    # 1. Target Time Series Plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    titles = [
        "Handysize (BDI HSI)",
        "Supramax (BDI SI)",
        "Panamax (BDI PI)",
        "Capesize (BDI CI)",
    ]
    for i, col in enumerate(TARGET_COLUMNS):
        if col in df.columns:
            axes[i].plot(dates, df[col], color=colors[i], lw=1.2, label=col)
            axes[i].set_ylabel("Index Level")
            axes[i].set_title(titles[i], loc="left", fontsize=11, fontweight="bold")
            axes[i].legend(loc="upper left")
    axes[-1].set_xlabel("Date")
    fig.suptitle("Baltic Dry Bulk Freight Sub-Indices (2012 - 2019)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    p1 = out_path / "01_target_time_series.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    saved_plots["target_time_series"] = str(p1)

    # 2. Normalized Comparative Targets Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, col in enumerate(TARGET_COLUMNS):
        if col in df.columns:
            s = df[col].dropna()
            # Normalize to 100 at start of series
            base = s.iloc[0]
            normalized = (df[col] / base) * 100
            ax.plot(dates, normalized, label=f"{col.upper()} (Base 100={base:.0f})", lw=1.3, color=colors[i])
    ax.set_title("Normalized Freight Trajectories Comparison (Base = 100 on 2012-08-01)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalized Index (Base=100)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    plt.tight_layout()
    p2 = out_path / "02_normalized_targets_comparison.png"
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    saved_plots["normalized_comparison"] = str(p2)

    # 3. Correlation Heatmap
    corr_cols = [
        *TARGET_COLUMNS,
        *MARKET_COLUMNS,
        "gpr",
        "gpr_acts",
        "gpr_threats",
        "wind_speed_max_kmh",
        "pressure_hpa",
    ]
    valid_corr_cols = [c for c in corr_cols if c in df.columns]
    corr_mat = df[valid_corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap: Freight Targets & Exogenous Drivers", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p3 = out_path / "03_correlation_heatmap.png"
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    saved_plots["correlation_heatmap"] = str(p3)

    # 4. Target Distributions & Boxplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for i, col in enumerate(TARGET_COLUMNS):
        if col in df.columns:
            s = df[col].dropna()
            sns.histplot(s, kde=True, ax=axes[0, i], color=colors[i], bins=30)
            axes[0, i].set_title(f"{col.upper()} Distribution", fontweight="bold")
            axes[0, i].set_xlabel("Index Level")

            sns.boxplot(y=s, ax=axes[1, i], color=colors[i])
            axes[1, i].set_title(f"{col.upper()} Boxplot", fontweight="bold")
            axes[1, i].set_ylabel("Index Level")
    fig.suptitle("Freight Sub-Indices Value Distributions and Dispersion", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p4 = out_path / "04_target_distributions.png"
    fig.savefig(p4, dpi=200)
    plt.close(fig)
    saved_plots["target_distributions"] = str(p4)

    # 5. Volatility (Day-to-Day % Change Distributions)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, col in enumerate(TARGET_COLUMNS):
        if col in df.columns:
            pct_change = df[col].dropna().pct_change().dropna() * 100
            sns.histplot(pct_change, kde=True, ax=axes[i], color=colors[i], bins=40)
            axes[i].set_title(f"{col.upper()} Daily Return (%) Distribution", fontweight="bold")
            axes[i].set_xlabel("Daily Change (%)")
            axes[i].axvline(0, color="black", linestyle="--", lw=1)
    fig.suptitle("Day-to-Day Freight Rate Volatility (%) on Trading Days", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p5 = out_path / "05_volatility_distributions.png"
    fig.savefig(p5, dpi=200)
    plt.close(fig)
    saved_plots["volatility_distributions"] = str(p5)

    # 6. Autocorrelation Function (ACF) Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, col in enumerate(TARGET_COLUMNS):
        if col in df.columns:
            acf_res = compute_autocorrelation(df[col].dropna(), nlags=30)
            lags = acf_res["lags"]
            acf_vals = acf_res["acf"]
            axes[i].stem(lags, acf_vals)
            axes[i].axhline(0, color="black", lw=1)
            axes[i].axhline(1.96 / np.sqrt(len(df[col].dropna())), color="red", linestyle="--", lw=0.8)
            axes[i].axhline(-1.96 / np.sqrt(len(df[col].dropna())), color="red", linestyle="--", lw=0.8)
            axes[i].set_title(f"{col.upper()} Autocorrelation (ACF)", fontweight="bold")
            axes[i].set_xlabel("Lag (Trading Days)")
            axes[i].set_ylabel("Autocorrelation")
    fig.suptitle("Autocorrelation Function (ACF) Across Freight Targets (Lags 1-30)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p6 = out_path / "06_autocorrelation_acf.png"
    fig.savefig(p6, dpi=200)
    plt.close(fig)
    saved_plots["acf_plots"] = str(p6)

    return saved_plots
