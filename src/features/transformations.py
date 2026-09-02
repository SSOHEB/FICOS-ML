"""Causal transformations: differences, percentage changes, and returns."""

from typing import List
import numpy as np
import pandas as pd


def create_differences(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [1, 5, 21],
) -> pd.DataFrame:
    """Generate causal historical differences: X_t - X_{t-k}.

    Args:
        df: Input DataFrame.
        columns: Column names to difference.
        windows: List of positive integer lookback steps k (k >= 1).

    Returns:
        pd.DataFrame: DataFrame containing difference features.
    """
    diff_features = {}
    for col in columns:
        if col not in df.columns:
            continue
        for k in windows:
            if k < 1:
                raise ValueError(f"Difference step must be >= 1, got {k}")
            feat_name = f"{col}_diff_{k}"
            diff_features[feat_name] = df[col].diff(periods=k)

    return pd.DataFrame(diff_features, index=df.index)


def create_percentage_changes(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [1, 5, 21],
) -> pd.DataFrame:
    """Generate causal historical percentage changes: (X_t - X_{t-k}) / X_{t-k} * 100.

    Args:
        df: Input DataFrame.
        columns: Column names to transform.
        windows: List of positive integer lookback steps k (k >= 1).

    Returns:
        pd.DataFrame: DataFrame containing percentage change features.
    """
    pct_features = {}
    for col in columns:
        if col not in df.columns:
            continue
        for k in windows:
            if k < 1:
                raise ValueError(f"Percentage change step must be >= 1, got {k}")
            feat_name = f"{col}_pct_change_{k}"
            pct_s = df[col].pct_change(periods=k) * 100.0
            pct_features[feat_name] = pct_s.replace([np.inf, -np.inf], np.nan)

    return pd.DataFrame(pct_features, index=df.index)


def create_log_returns(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [1, 5],
) -> pd.DataFrame:
    """Generate causal log returns: ln(X_t / X_{t-k}).

    Args:
        df: Input DataFrame.
        columns: Column names.
        windows: List of lookback steps.

    Returns:
        pd.DataFrame: DataFrame containing log return features.
    """
    log_features = {}
    for col in columns:
        if col not in df.columns:
            continue
        for k in windows:
            if k < 1:
                raise ValueError(f"Log return step must be >= 1, got {k}")
            feat_name = f"{col}_log_return_{k}"
            ratio = df[col] / df[col].shift(k)
            # Safe log transformation for strictly positive market values
            log_features[feat_name] = np.log(ratio.where(ratio > 0))

    return pd.DataFrame(log_features, index=df.index)
