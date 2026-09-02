"""Causal lag feature generation for freight time series."""

from typing import List, Dict, Optional
import pandas as pd


def create_autoregressive_lags(
    df: pd.DataFrame,
    target_cols: List[str],
    lags: List[int] = [1, 2, 3, 5, 10, 21],
) -> pd.DataFrame:
    """Generate strictly causal autoregressive lag features.

    Args:
        df: Input DataFrame indexed by trading day.
        target_cols: Target column names.
        lags: List of integer lag steps (must all be >= 1).

    Returns:
        pd.DataFrame: DataFrame containing generated lag features.
    """
    for lag in lags:
        if lag < 1:
            raise ValueError(f"Lag must be >= 1 to prevent data leakage, got {lag}")

    lag_features = {}
    for col in target_cols:
        if col not in df.columns:
            continue
        for lag in lags:
            feat_name = f"{col}_lag_{lag}"
            lag_features[feat_name] = df[col].shift(lag)

    return pd.DataFrame(lag_features, index=df.index)


def create_cross_vessel_lags(
    df: pd.DataFrame,
    cross_map: Dict[str, List[str]],
    lags: List[int] = [1, 5],
) -> pd.DataFrame:
    """Generate strictly causal cross-vessel lag features.

    Args:
        df: Input DataFrame.
        cross_map: Mapping of target to list of cross-segment predictors.
        lags: List of positive integer lag steps.

    Returns:
        pd.DataFrame: DataFrame containing generated cross-vessel features.
    """
    for lag in lags:
        if lag < 1:
            raise ValueError(f"Cross lag must be >= 1 to prevent data leakage, got {lag}")

    cross_features = {}
    created_pairs = set()

    for target, sources in cross_map.items():
        for source in sources:
            if source not in df.columns:
                continue
            for lag in lags:
                pair_key = (source, lag)
                if pair_key in created_pairs:
                    continue
                created_pairs.add(pair_key)
                feat_name = f"cross_{source}_lag_{lag}"
                cross_features[feat_name] = df[source].shift(lag)

    return pd.DataFrame(cross_features, index=df.index)
