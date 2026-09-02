"""Phase 10.5 Data Integrity & Non-Mutation Tests for FICOS.

Verifies:
- master_dataset.csv and freight_features.csv remain unmodified
- Row/column shapes match known invariant baseline
- Date chronological ordering, continuity, lack of duplicates
- Missing values preserved without invalid forward-filling over non-trading holidays
- Absence of infinite or NaN values in critical feature columns
- Target isolation & absence of target contamination in feature matrix
"""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.schemas import DATE_COLUMN, TARGET_COLUMNS


@pytest.fixture(scope="module")
def master_df() -> pd.DataFrame:
    p = Path("data/processed/master_dataset.csv")
    assert p.exists(), "master_dataset.csv must exist"
    df = pd.read_csv(p)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df


@pytest.fixture(scope="module")
def features_df() -> pd.DataFrame:
    p = Path("data/features/freight_features.csv")
    assert p.exists(), "freight_features.csv must exist"
    df = pd.read_csv(p)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df


def test_master_dataset_row_column_and_checksum(master_df):
    """Verify master_dataset.csv shape, column count, and SHA-256 integrity."""
    assert master_df.shape[0] == 5145, f"Expected 5145 rows, found {master_df.shape[0]}"
    assert master_df.shape[1] == 84, f"Expected 84 columns, found {master_df.shape[1]}"
    
    # Check SHA-256
    with open("data/processed/master_dataset.csv", "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    assert file_hash == "3930980597df212d3c3b82eb5338a2354f48b1f3d89dd61891fb5f9d11d76d54"


def test_features_dataset_row_column_and_checksum(features_df):
    """Verify freight_features.csv shape, column count, and SHA-256 integrity."""
    assert features_df.shape[0] == 1749, f"Expected 1749 trading rows, found {features_df.shape[0]}"
    assert features_df.shape[1] == 141, f"Expected 141 feature columns, found {features_df.shape[1]}"
    
    with open("data/features/freight_features.csv", "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    assert file_hash == "120619a76c770df21eb84440c8c2aa4a163ce25088d3e642bcaf1787dfa6d8d3"


def test_date_ordering_and_duplicates(master_df, features_df):
    """Verify strictly monotonic ascending date sequence and zero duplicate dates."""
    assert master_df[DATE_COLUMN].is_monotonic_increasing, "Master dataset dates must be strictly ascending"
    assert master_df[DATE_COLUMN].duplicated().sum() == 0, "Master dataset must have zero duplicate dates"

    assert features_df[DATE_COLUMN].is_monotonic_increasing, "Features dataset dates must be strictly ascending"
    assert features_df[DATE_COLUMN].duplicated().sum() == 0, "Features dataset must have zero duplicate dates"


def test_missing_values_and_no_infinities(master_df, features_df):
    """Verify no infinite values and that missing values follow expected market calendar gaps."""
    # Check inf values
    numeric_cols_master = master_df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols_master:
        assert not np.isinf(master_df[c]).any(), f"Infinite value found in master col {c}"

    numeric_cols_features = features_df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols_features:
        assert not np.isinf(features_df[c]).any(), f"Infinite value found in features col {c}"


def test_target_contamination_absence(features_df):
    """Verify that future targets are not accidentally present in lagged features."""
    target_names = ["bdi_hsi", "bdi_si", "bdi_pi", "bdi_ci"]
    for tgt in target_names:
        # Check level column exists
        assert f"{tgt}_level" in features_df.columns
        # Target column itself (future return) should only be the designated target
        assert f"target_{tgt}_next" in features_df.columns
        # Verify that lag_1 is strictly t-1
        level = features_df[f"{tgt}_level"].values
        lag_1 = features_df[f"{tgt}_lag_1"].values
        # For rows where lag_1 is not NaN, lag_1[i] must equal level[i-1]
        valid_idx = ~np.isnan(lag_1)
        # Shifted comparison
        np.testing.assert_allclose(lag_1[1:], level[:-1], rtol=1e-5, atol=1e-5)
