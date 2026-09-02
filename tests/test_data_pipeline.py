"""Tests for Phase 2 data loaders, validators, cleaners, and pipeline."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.schemas import (
    DATE_COLUMN,
    TARGET_COLUMNS,
    MARKET_COLUMNS,
    GPR_COLUMNS,
    WEATHER_COLUMNS,
    FLAG_COLUMNS,
    ALL_EXPECTED_COLUMNS,
)
from src.data.loaders import load_raw_dataset, load_yaml_config
from src.data.validators import validate_dataset, ValidationReport
from src.data.cleaners import clean_master_data
from src.data.pipeline import run_pipeline


@pytest.fixture
def synthetic_raw_df() -> pd.DataFrame:
    """Fixture providing a synthetic dataset with valid schema and intentional unsorted/duplicate dates."""
    dates = [
        "2020-01-03",
        "2020-01-01",
        "2020-01-02",
        "2020-01-02",  # intentional duplicate date
        "2020-01-04",
    ]
    data = {
        "date": dates,
        "bdi_hsi": [500.0, 510.0, 520.0, 520.0, np.nan],
        "bdi_si": [900.0, 910.0, 920.0, 920.0, np.nan],
        "bdi_pi": [800.0, 810.0, 820.0, 820.0, np.nan],
        "bdi_ci": [1100.0, 1110.0, 1120.0, 1120.0, np.nan],
        "wti_usd_bbl": [60.0, 61.0, 62.0, 62.0, np.nan],
        "brent_usd_bbl": [65.0, 66.0, 67.0, 67.0, np.nan],
        "usd_inr": [71.0, 71.2, 71.5, 71.5, np.nan],
        "gpr": [100.0, 102.0, 105.0, 105.0, 98.0],
        "gpr_acts": [50.0, 52.0, 55.0, 55.0, 48.0],
        "gpr_threats": [80.0, 82.0, 85.0, 85.0, 78.0],
        "gpr_ma7": [95.0, 97.0, 99.0, 99.0, 96.0],
        "gpr_ma30": [90.0, 91.0, 92.0, 92.0, 93.0],
        "gpr_article_count": [700, 710, 720, 720, 690],
        "wind_speed_max_kmh": [15.0, 16.0, 17.0, 17.0, 14.0],
        "wind_gust_max_kmh": [30.0, 32.0, 35.0, 35.0, 28.0],
        "precip_mm": [0.0, 1.2, 0.0, 0.0, 5.0],
        "pressure_hpa": [1012.0, 1011.5, 1010.0, 1010.0, 1009.0],
        "is_bdi_trading_day": [True, True, True, True, False],
        "is_market_trading_day": [True, True, True, True, False],
    }
    return pd.DataFrame(data)


def test_schema_definitions():
    """Verify schema column completeness."""
    assert DATE_COLUMN in ALL_EXPECTED_COLUMNS
    for tgt in TARGET_COLUMNS:
        assert tgt in ALL_EXPECTED_COLUMNS
    for mkt in MARKET_COLUMNS:
        assert mkt in ALL_EXPECTED_COLUMNS
    for gpr in GPR_COLUMNS:
        assert gpr in ALL_EXPECTED_COLUMNS
    for weather in WEATHER_COLUMNS:
        assert weather in ALL_EXPECTED_COLUMNS
    for flag in FLAG_COLUMNS:
        assert flag in ALL_EXPECTED_COLUMNS


def test_date_parsing_and_loading(tmp_path, synthetic_raw_df):
    """Test explicit date parsing during loading."""
    csv_file = tmp_path / "test_raw.csv"
    synthetic_raw_df.to_csv(csv_file, index=False)

    df_loaded = load_raw_dataset(csv_file)
    assert pd.api.types.is_datetime64_any_dtype(df_loaded["date"])
    assert len(df_loaded) == 5


def test_duplicate_detection(synthetic_raw_df):
    """Test that validator detects duplicate dates."""
    synthetic_raw_df["date"] = pd.to_datetime(synthetic_raw_df["date"])
    report = validate_dataset(synthetic_raw_df)
    assert report.duplicate_dates_count == 1
    assert not report.is_valid  # Fails due to duplicate date and unsorted order


def test_chronological_ordering_and_cleaning(synthetic_raw_df):
    """Test that cleaners sort dates ascending and deduplicate."""
    synthetic_raw_df["date"] = pd.to_datetime(synthetic_raw_df["date"])
    cleaned = clean_master_data(synthetic_raw_df)

    # 1. Duplicates removed (was 5 rows, should be 4 unique dates)
    assert len(cleaned) == 4
    assert cleaned["date"].duplicated().sum() == 0

    # 2. Chronological order verified
    assert cleaned["date"].is_monotonic_increasing
    assert cleaned["date"].iloc[0] == pd.Timestamp("2020-01-01")
    assert cleaned["date"].iloc[-1] == pd.Timestamp("2020-01-04")


def test_missing_values_preservation(synthetic_raw_df):
    """Confirm that missing target values on non-trading days are strictly preserved as NaN."""
    synthetic_raw_df["date"] = pd.to_datetime(synthetic_raw_df["date"])
    cleaned = clean_master_data(synthetic_raw_df)

    # Non-trading day (2020-01-04) must remain NaN
    non_trading_row = cleaned[cleaned["date"] == "2020-01-04"].iloc[0]
    for tgt in TARGET_COLUMNS:
        assert pd.isna(non_trading_row[tgt]), f"Target {tgt} should be NaN for non-trading day"


def test_full_pipeline_execution(tmp_path):
    """Test full pipeline execution with real raw file and custom output path."""
    raw_path = Path("data/raw/master_daily.csv")
    if not raw_path.exists():
        pytest.skip("Raw file not present in data/raw")

    # Create temporary config
    cfg_file = tmp_path / "test_config.yaml"
    out_file = tmp_path / "processed_master.csv"
    cfg_content = f"""
raw:
  dir: "data/raw"
  filename: "master_daily.csv"
processed:
  dir: "{tmp_path.as_posix()}"
  filename: "processed_master.csv"
"""
    cfg_file.write_text(cfg_content)

    df_out = run_pipeline(cfg_file)
    assert out_file.exists()
    assert len(df_out) == 2556
    assert df_out[DATE_COLUMN].is_monotonic_increasing
    assert df_out[DATE_COLUMN].duplicated().sum() == 0
