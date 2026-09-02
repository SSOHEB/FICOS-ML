"""Executable data pipeline for constructing and saving the master processed dataset."""

from pathlib import Path
from typing import Optional, Union
import sys
import pandas as pd

from src.data.loaders import load_yaml_config, load_raw_dataset
from src.data.validators import validate_dataset, ValidationReport
from src.data.cleaners import clean_master_data
from src.data.schemas import DATE_COLUMN, TARGET_COLUMNS


def run_pipeline(config_path: Union[str, Path] = "configs/data.yaml") -> pd.DataFrame:
    """Execute the Phase 2 data pipeline: load, validate, clean, and export.

    Args:
        config_path: Path to configuration YAML file.

    Returns:
        pd.DataFrame: The validated and processed master DataFrame.
    """
    print("=" * 65)
    print("  FICOS - Data Pipeline (Phase 2)")
    print("  Building Master Historical Dataset")
    print("=" * 65)

    # 1. Load configuration
    config = load_yaml_config(config_path)
    raw_dir = Path(config.get("raw", {}).get("dir", "data/raw"))
    raw_filename = config.get("raw", {}).get("filename", "master_daily.csv")
    raw_path = raw_dir / raw_filename

    proc_dir = Path(config.get("processed", {}).get("dir", "data/processed"))
    proc_filename = config.get("processed", {}).get("filename", "master_dataset.csv")
    proc_path = proc_dir / proc_filename

    print(f"\n[1] Source Configuration:")
    print(f"    - Raw Data File:       {raw_path}")
    print(f"    - Processed Output:    {proc_path}")

    # 2. Load raw dataset
    print(f"\n[2] Loading Raw Dataset...")
    df_raw = load_raw_dataset(raw_path)
    print(f"    Loaded {len(df_raw):,} records with {df_raw.shape[1]} columns.")

    # 3. Validate raw data
    print(f"\n[3] Validating Raw Dataset...")
    raw_report: ValidationReport = validate_dataset(df_raw)
    print(f"    Status: [{'PASSED' if raw_report.is_valid else 'FAILED'}]")
    if not raw_report.is_valid:
        for err in raw_report.errors:
            print(f"    [ERROR] {err}")
        raise ValueError("Raw dataset failed validation checks.")

    # 4. Clean and normalize
    print(f"\n[4] Cleaning and Normalizing Dataset...")
    df_clean = clean_master_data(df_raw)
    print(f"    Cleaned records: {len(df_clean):,} | Duplicate dates: {df_clean[DATE_COLUMN].duplicated().sum()}")

    # 5. Validate processed dataset
    print(f"\n[5] Validating Processed Master Dataset...")
    clean_report: ValidationReport = validate_dataset(df_clean)
    print(f"    Status: [{'PASSED' if clean_report.is_valid else 'FAILED'}]")
    if not clean_report.is_valid:
        for err in clean_report.errors:
            print(f"    [ERROR] {err}")
        raise ValueError("Processed dataset failed validation checks.")

    # 6. Save master dataset
    proc_dir.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(proc_path, index=False)
    print(f"\n[6] Export:")
    print(f"    Successfully written master dataset to: {proc_path.resolve()}")

    # 7. Print Concise Summary
    date_min = df_clean[DATE_COLUMN].min().strftime("%Y-%m-%d")
    date_max = df_clean[DATE_COLUMN].max().strftime("%Y-%m-%d")
    dup_count = df_clean[DATE_COLUMN].duplicated().sum()

    print("\n" + "=" * 65)
    print("  MASTER DATASET SUMMARY:")
    print("=" * 65)
    print(f"  * Total Rows:            {len(df_clean):,}")
    print(f"  * Total Columns:         {df_clean.shape[1]}")
    print(f"  * Date Range:            {date_min} to {date_max}")
    print(f"  * Total Calendar Days:   {(pd.to_datetime(date_max) - pd.to_datetime(date_min)).days + 1:,}")
    print(f"  * Duplicate Dates:       {dup_count}")
    if "is_bdi_trading_day" in df_clean.columns:
        trading_days = int(df_clean["is_bdi_trading_day"].sum())
        non_trading = len(df_clean) - trading_days
        print(f"  * Trading Days:          {trading_days:,} (active freight series)")
        print(f"  * Non-Trading Days:      {non_trading:,} (weekends / market holidays)")

    print(f"\n  * Missing Values Count:")
    for col, count in clean_report.null_counts.items():
        pct = (count / len(df_clean)) * 100
        print(f"      - {col:<22}: {count:>5} ({pct:.1f}%)")

    print("\n  STATUS: [SUCCESS] Phase 2 Data Pipeline Execution Complete.")
    print("=" * 65 + "\n")

    return df_clean


def main():
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/data.yaml"
    try:
        run_pipeline(config_file)
    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
