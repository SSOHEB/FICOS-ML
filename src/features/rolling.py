"""Causal rolling window feature generation."""

from typing import List, Optional
import numpy as np
import pandas as pd


def create_rolling_statistics(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [7, 30],
    statistics: List[str] = ["mean", "std", "min", "max"],
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Compute backward-looking (strictly causal) rolling window statistics.

    Args:
        df: Input DataFrame.
        columns: Column names to compute statistics for.
        windows: List of integer window sizes (e.g. 7, 30).
        statistics: List of statistics to compute ('mean', 'std', 'min', 'max').
        min_periods: Minimum observations required in window. Defaults to half the window size.

    Returns:
        pd.DataFrame: DataFrame containing rolling statistics.
    """
    rolling_features = {}

    for col in columns:
        if col not in df.columns:
            continue
        for w in windows:
            if w < 2:
                raise ValueError(f"Rolling window must be >= 2, got {w}")
            
            req_periods = min_periods if min_periods is not None else max(1, w // 2)
            # strictly backward-looking: center=False
            roll = df[col].rolling(window=w, min_periods=req_periods, center=False)

            if "mean" in statistics:
                rolling_features[f"{col}_roll_mean_{w}"] = roll.mean()
            if "std" in statistics:
                rolling_features[f"{col}_roll_std_{w}"] = roll.std()
            if "min" in statistics:
                rolling_features[f"{col}_roll_min_{w}"] = roll.min()
            if "max" in statistics:
                rolling_features[f"{col}_roll_max_{w}"] = roll.max()

    return pd.DataFrame(rolling_features, index=df.index)


def create_rolling_volatility(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [7, 30],
) -> pd.DataFrame:
    """Compute backward-looking rolling return volatility (std of daily pct changes).

    Args:
        df: Input DataFrame.
        columns: Target column names.
        windows: List of window sizes.

    Returns:
        pd.DataFrame: DataFrame containing rolling return volatility features.
    """
    vol_features = {}

    for col in columns:
        if col not in df.columns:
            continue
        # Causal daily percent return
        daily_pct = df[col].pct_change(periods=1) * 100.0
        
        for w in windows:
            req_periods = max(2, w // 2)
            vol_features[f"{col}_return_vol_{w}"] = daily_pct.rolling(
                window=w, min_periods=req_periods, center=False
            ).std()

    return pd.DataFrame(vol_features, index=df.index)
