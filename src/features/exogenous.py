"""Causal exogenous (macro, energy, FX, geopolitical risk, weather, calendar) features."""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.data.schemas import (
    DATE_COLUMN,
    MARKET_COLUMNS,
    GPR_COLUMNS,
    WEATHER_COLUMNS,
)


def create_macro_features(
    df: pd.DataFrame,
    energy_cols: List[str] = ["wti_usd_bbl", "brent_usd_bbl"],
    fx_cols: List[str] = ["usd_inr"],
    lags: List[int] = [1],
    pct_windows: List[int] = [1, 5],
) -> pd.DataFrame:
    """Generate causal features from energy and foreign exchange series.

    Args:
        df: Input DataFrame.
        energy_cols: Energy price column names.
        fx_cols: Foreign exchange column names.
        lags: Lag lookbacks (must all be >= 1).
        pct_windows: Return lookback windows.

    Returns:
        pd.DataFrame: DataFrame of engineered macro features.
    """
    macro_features = {}
    cols = energy_cols + fx_cols

    for col in cols:
        if col not in df.columns:
            continue
        
        for lag in lags:
            if lag < 1:
                raise ValueError(f"Lag must be >= 1, got {lag}")
            macro_features[f"{col}_lag_{lag}"] = df[col].shift(lag)

        for w in pct_windows:
            if w < 1:
                raise ValueError(f"Window must be >= 1, got {w}")
            # Causal return of the lagged price
            pct_s = df[col].pct_change(periods=w) * 100.0
            macro_features[f"{col}_pct_change_{w}"] = pct_s.replace([np.inf, -np.inf], np.nan)

    return pd.DataFrame(macro_features, index=df.index)


def create_geopolitical_features(
    df: pd.DataFrame,
    gpr_cols: List[str] = ["gpr", "gpr_acts", "gpr_threats"],
    lags: List[int] = [1],
) -> pd.DataFrame:
    """Generate causal geopolitical risk and shock indicators.

    Args:
        df: Input DataFrame.
        gpr_cols: GPR column names.
        lags: Lag steps.

    Returns:
        pd.DataFrame: DataFrame of geopolitical features.
    """
    gpr_features = {}

    for col in gpr_cols:
        if col not in df.columns:
            continue
        for lag in lags:
            if lag < 1:
                raise ValueError(f"Lag must be >= 1, got {lag}")
            gpr_features[f"{col}_lag_{lag}"] = df[col].shift(lag)

        # 1-day differences and percentage changes
        gpr_features[f"{col}_diff_1"] = df[col].diff(periods=1)
        pct_s = df[col].pct_change(periods=1) * 100.0
        gpr_features[f"{col}_pct_change_1"] = pct_s.replace([np.inf, -np.inf], np.nan)

    # Shock indicators relative to baseline moving averages
    if "gpr" in df.columns and "gpr_ma30" in df.columns:
        # Causal spike ratio: gpr / gpr_ma30
        ratio = df["gpr"] / (df["gpr_ma30"].replace(0, np.nan))
        gpr_features["gpr_spike_ratio_ma30"] = ratio.replace([np.inf, -np.inf], np.nan)
    if "gpr" in df.columns and "gpr_ma7" in df.columns:
        ratio7 = df["gpr"] / (df["gpr_ma7"].replace(0, np.nan))
        gpr_features["gpr_spike_ratio_ma7"] = ratio7.replace([np.inf, -np.inf], np.nan)

    return pd.DataFrame(gpr_features, index=df.index)


def create_weather_features(
    df: pd.DataFrame,
    weather_cols: List[str] = ["wind_speed_max_kmh", "precip_mm", "pressure_hpa"],
    lags: List[int] = [1],
) -> pd.DataFrame:
    """Generate causal weather features.

    Args:
        df: Input DataFrame.
        weather_cols: Weather column names.
        lags: Lag steps.

    Returns:
        pd.DataFrame: DataFrame of weather features.
    """
    weather_features = {}

    for col in weather_cols:
        if col not in df.columns:
            continue
        for lag in lags:
            if lag < 1:
                raise ValueError(f"Lag must be >= 1, got {lag}")
            weather_features[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # Barometric pressure daily change
    if "pressure_hpa" in df.columns:
        weather_features["pressure_hpa_diff_1"] = df["pressure_hpa"].diff(periods=1)

    return pd.DataFrame(weather_features, index=df.index)


def create_calendar_features(
    df: pd.DataFrame,
    date_col: str = DATE_COLUMN,
    features: List[str] = ["month", "quarter", "day_of_week"],
) -> pd.DataFrame:
    """Extract deterministic calendar features from the date column.

    Args:
        df: Input DataFrame.
        date_col: Name of date column.
        features: List of calendar features to extract.

    Returns:
        pd.DataFrame: DataFrame containing calendar features.
    """
    if date_col not in df.columns:
        return pd.DataFrame(index=df.index)

    dates = pd.to_datetime(df[date_col])
    cal_features = {}

    if "month" in features:
        cal_features["cal_month"] = dates.dt.month.astype(int)
    if "quarter" in features:
        cal_features["cal_quarter"] = dates.dt.quarter.astype(int)
    if "day_of_week" in features:
        cal_features["cal_day_of_week"] = dates.dt.dayofweek.astype(int)
    if "day_of_month" in features:
        cal_features["cal_day_of_month"] = dates.dt.day.astype(int)
    if "is_month_end" in features:
        cal_features["cal_is_month_end"] = dates.dt.is_month_end.astype(int)

    return pd.DataFrame(cal_features, index=df.index)
