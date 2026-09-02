"""Data cleaning and normalization routines for FICOS ML."""

import pandas as pd
from typing import List

from src.data.schemas import (
    DATE_COLUMN,
    ALL_EXPECTED_COLUMNS,
    TARGET_COLUMNS,
    NUMERIC_COLUMNS,
)


def clean_master_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the dataset into a master historical series.

    Guarantees:
    - Chronological ascending ordering by date.
    - No duplicate dates or rows.
    - Explicit missing values preserved (NO forward-fill or interpolation of target series).
    - Standardized column ordering.

    Args:
        df: Input raw DataFrame.

    Returns:
        pd.DataFrame: Cleaned and normalized master DataFrame.
    """
    clean_df = df.copy()

    # 1. Ensure date is datetime64 and normalized (no time component)
    clean_df[DATE_COLUMN] = pd.to_datetime(clean_df[DATE_COLUMN]).dt.normalize()

    # 2. Remove exact duplicate rows if any exist
    clean_df = clean_df.drop_duplicates()

    # 3. Deduplicate by date keeping first record if duplicate dates exist
    if clean_df[DATE_COLUMN].duplicated().any():
        clean_df = clean_df.drop_duplicates(subset=[DATE_COLUMN], keep="first")

    # 4. Sort chronologically ascending
    clean_df = clean_df.sort_values(by=DATE_COLUMN, ascending=True).reset_index(drop=True)

    # 5. Ensure column order matches schema specification where present
    ordered_cols: List[str] = [c for c in ALL_EXPECTED_COLUMNS if c in clean_df.columns]
    # append any additional unexpected columns at the end
    extra_cols: List[str] = [c for c in clean_df.columns if c not in ordered_cols]
    clean_df = clean_df[ordered_cols + extra_cols]

    return clean_df
