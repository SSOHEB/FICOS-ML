"""Unit and integration tests for Phase 3 Exploratory Data Analysis functions."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.evaluation.eda import (
    profile_dataset,
    compute_target_statistics,
    compute_correlation_matrix,
    analyze_missingness,
    detect_outliers,
    compute_volatility_metrics,
    evaluate_stationarity,
    compute_autocorrelation,
    generate_eda_plots,
)


@pytest.fixture
def synthetic_eda_df() -> pd.DataFrame:
    """Fixture providing a clean synthetic time series dataset for EDA testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    
    # Simulate trading day flag (70 trading days, 30 weekends/holidays)
    is_trading = [d.weekday() < 5 for d in dates]
    
    # Generate random walk target
    base = 500.0
    hsi = []
    val = base
    for t in is_trading:
        if t:
            val += np.random.normal(0, 5)
            hsi.append(val)
        else:
            hsi.append(np.nan)

    data = {
        "date": dates,
        "bdi_hsi": hsi,
        "bdi_si": [x * 1.8 if not np.isnan(x) else np.nan for x in hsi],
        "bdi_pi": [x * 2.2 if not np.isnan(x) else np.nan for x in hsi],
        "bdi_ci": [x * 3.5 if not np.isnan(x) else np.nan for x in hsi],
        "wti_usd_bbl": np.random.uniform(50, 70, size=100),
        "brent_usd_bbl": np.random.uniform(55, 75, size=100),
        "usd_inr": np.random.uniform(70, 75, size=100),
        "gpr": np.random.uniform(80, 120, size=100),
        "is_bdi_trading_day": is_trading,
        "is_market_trading_day": is_trading,
    }
    return pd.DataFrame(data)


def test_profile_dataset(synthetic_eda_df):
    """Test dataset profiling metrics extraction."""
    profile = profile_dataset(synthetic_eda_df)
    assert profile["total_rows"] == 100
    assert profile["total_columns"] == 11
    assert profile["trading_days"] == sum(synthetic_eda_df["is_bdi_trading_day"])
    assert profile["non_trading_days"] == 100 - profile["trading_days"]
    assert profile["duplicate_dates"] == 0


def test_compute_target_statistics(synthetic_eda_df):
    """Test target statistical aggregation."""
    stats_df = compute_target_statistics(synthetic_eda_df, ["bdi_hsi", "bdi_si"])
    assert len(stats_df) == 2
    assert "mean" in stats_df.columns
    assert "cv_pct" in stats_df.columns
    assert stats_df.loc[stats_df["target"] == "bdi_hsi", "count"].iloc[0] == synthetic_eda_df["bdi_hsi"].count()


def test_compute_correlation_matrix(synthetic_eda_df):
    """Test correlation computation."""
    corr = compute_correlation_matrix(synthetic_eda_df, ["bdi_hsi", "bdi_si", "wti_usd_bbl"])
    assert corr.shape == (3, 3)
    assert np.isclose(corr.loc["bdi_hsi", "bdi_hsi"], 1.0)
    assert np.isclose(corr.loc["bdi_hsi", "bdi_si"], 1.0)  # perfect linear relation in synthetic fixture


def test_analyze_missingness(synthetic_eda_df):
    """Test missingness analysis and trading-day anomaly detection."""
    miss_df = analyze_missingness(synthetic_eda_df)
    assert len(miss_df) == len(synthetic_eda_df.columns)
    hsi_row = miss_df[miss_df["column"] == "bdi_hsi"].iloc[0]
    assert hsi_row["nulls_trading_days"] == 0
    assert not hsi_row["has_trading_anomalies"]


def test_detect_outliers():
    """Test outlier detection with IQR and z-score methods."""
    data = pd.Series([10.0, 11.0, 10.5, 10.2, 9.8, 10.1, 100.0])  # 100.0 is an outlier
    out_iqr = detect_outliers(data, method="iqr", threshold=1.5)
    assert out_iqr["outlier_count"] == 1
    assert out_iqr["max_outlier"] == 100.0

    out_z = detect_outliers(data, method="zscore", threshold=2.0)
    assert out_z["outlier_count"] >= 1


def test_compute_volatility_metrics():
    """Test return and volatility computation."""
    data = pd.Series([100.0, 105.0, 102.0, 110.0, 108.0])
    vol = compute_volatility_metrics(data)
    assert vol["n_obs"] == 4
    assert "annualized_volatility_pct" in vol
    assert vol["annualized_volatility_pct"] > 0


def test_stationarity_evaluation():
    """Test stationarity test wrapper on synthetic stationary and non-stationary series."""
    np.random.seed(42)
    # White noise (Stationary)
    stationary_series = pd.Series(np.random.normal(0, 1, size=100))
    res_adf = evaluate_stationarity(stationary_series, test_type="adf")
    assert res_adf["p_value"] < 0.05
    assert res_adf["is_stationary_5pct"] is True

    res_kpss = evaluate_stationarity(stationary_series, test_type="kpss")
    assert res_kpss["is_stationary_5pct"] is True


def test_compute_autocorrelation():
    """Test ACF and PACF computation."""
    series = pd.Series(np.arange(100, dtype=float))
    acf_pacf = compute_autocorrelation(series, nlags=10)
    assert len(acf_pacf["acf"]) == 11
    assert acf_pacf["acf"][0] == 1.0


def test_generate_eda_plots(tmp_path, synthetic_eda_df):
    """Test that EDA plot generation creates valid image files."""
    plots = generate_eda_plots(synthetic_eda_df, output_dir=tmp_path)
    assert len(plots) == 6
    for name, path_str in plots.items():
        assert Path(path_str).exists()
        assert Path(path_str).stat().st_size > 1000  # valid image payload


def test_master_dataset_unmodified():
    """Verify data integrity: EDA checks must not modify data/processed/master_dataset.csv."""
    master_path = Path("data/processed/master_dataset.csv")
    if not master_path.exists():
        pytest.skip("Master dataset not created yet")

    mtime_before = master_path.stat().st_mtime
    df = pd.read_csv(master_path)
    _ = compute_target_statistics(df)
    _ = analyze_missingness(df)
    mtime_after = master_path.stat().st_mtime
    assert mtime_before == mtime_after, "Master dataset file was modified by EDA execution!"
