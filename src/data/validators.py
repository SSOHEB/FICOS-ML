"""Validation logic for the historical dataset."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd

from src.data.schemas import (
    DATE_COLUMN,
    ALL_EXPECTED_COLUMNS,
    TARGET_COLUMNS,
    NUMERIC_COLUMNS,
    FLAG_COLUMNS,
)


@dataclass
class ValidationReport:
    """Structured report produced during data validation."""
    is_valid: bool
    total_rows: int
    total_columns: int
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    is_chronological: bool = True
    duplicate_dates_count: int = 0
    missing_columns: List[str] = field(default_factory=list)
    null_counts: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"Validation Status: [{status}]",
            f"Rows: {self.total_rows:,} | Columns: {self.total_columns}",
            f"Date Range: {self.date_min} to {self.date_max}",
            f"Chronologically Ordered: {self.is_chronological}",
            f"Duplicate Dates: {self.duplicate_dates_count}",
        ]
        if self.missing_columns:
            lines.append(f"Missing Columns: {self.missing_columns}")
        if self.errors:
            lines.append("Errors:")
            for err in self.errors:
                lines.append(f"  - {err}")
        if self.warnings:
            lines.append("Warnings:")
            for warn in self.warnings:
                lines.append(f"  - {warn}")
        return "\n".join(lines)


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    """Validate DataFrame against structural, temporal, and completeness constraints.

    Args:
        df: Input DataFrame to validate.

    Returns:
        ValidationReport: Validation summary and findings.
    """
    errors: List[str] = []
    warnings: List[str] = []

    total_rows, total_columns = df.shape
    if total_rows == 0:
        errors.append("Dataset is completely empty.")
        return ValidationReport(
            is_valid=False,
            total_rows=0,
            total_columns=total_columns,
            errors=errors,
        )

    # 1. Column existence
    missing_cols = [c for c in ALL_EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing expected columns: {missing_cols}")

    # 2. Date column validation
    date_min = None
    date_max = None
    is_chronological = True
    duplicate_dates = 0

    if DATE_COLUMN in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[DATE_COLUMN]):
            errors.append(f"Column '{DATE_COLUMN}' is not datetime type (got {df[DATE_COLUMN].dtype}).")
        else:
            if df[DATE_COLUMN].isnull().any():
                errors.append(f"Column '{DATE_COLUMN}' contains null values ({df[DATE_COLUMN].isnull().sum()} nulls).")
            
            date_min = str(df[DATE_COLUMN].min().date())
            date_max = str(df[DATE_COLUMN].max().date())
            is_chronological = bool(df[DATE_COLUMN].is_monotonic_increasing)
            if not is_chronological:
                warnings.append(f"Dates are not sorted chronologically in ascending order.")

            duplicate_dates = int(df[DATE_COLUMN].duplicated().sum())
            if duplicate_dates > 0:
                errors.append(f"Found {duplicate_dates} duplicate date entries.")

    # 3. Numeric columns dtype validation
    for col in NUMERIC_COLUMNS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' is expected to be numeric, but found dtype {df[col].dtype}.")

    # 4. Null values reporting
    null_counts = {col: int(df[col].isnull().sum()) for col in df.columns if df[col].isnull().sum() > 0}

    # Verify that target nulls align with non-trading days if trading flags exist
    if "is_bdi_trading_day" in df.columns:
        for tgt in TARGET_COLUMNS:
            if tgt in df.columns:
                trading_missing = df[df["is_bdi_trading_day"]][tgt].isnull().sum()
                if trading_missing > 0:
                    warnings.append(f"Target '{tgt}' has {trading_missing} missing values on designated trading days.")

    is_valid = len(errors) == 0

    return ValidationReport(
        is_valid=is_valid,
        total_rows=total_rows,
        total_columns=total_columns,
        date_min=date_min,
        date_max=date_max,
        is_chronological=is_chronological,
        duplicate_dates_count=duplicate_dates,
        missing_columns=missing_cols,
        null_counts=null_counts,
        errors=errors,
        warnings=warnings,
    )
