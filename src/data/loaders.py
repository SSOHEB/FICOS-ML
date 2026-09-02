"""Data loaders for reading raw historical datasets."""

from pathlib import Path
from typing import Union, Dict, Any
import pandas as pd
import yaml

from src.data.schemas import DATE_COLUMN, ALL_EXPECTED_COLUMNS, NUMERIC_COLUMNS, BOOLEAN_COLUMNS


def load_yaml_config(config_path: Union[str, Path] = "configs/data.yaml") -> Dict[str, Any]:
    """Load and parse the YAML data pipeline configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_raw_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """Safely load raw historical CSV dataset, parsing dates and validating basic structure.

    Args:
        file_path: Path to the raw CSV file.

    Returns:
        pd.DataFrame: Loaded and typed DataFrame.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset file does not exist: {path.resolve()}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV from {path}: {e}") from e

    if df.empty:
        raise ValueError(f"Dataset at {path} is empty.")

    if DATE_COLUMN not in df.columns:
        raise KeyError(f"Expected date column '{DATE_COLUMN}' not found in {path}. Columns present: {list(df.columns)}")

    # Parse dates explicitly
    try:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format="%Y-%m-%d")
    except Exception:
        # Fallback to general ISO parsing
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    # Convert numeric columns explicitly without corrupting NaNs
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert boolean flags explicitly
    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    return df
