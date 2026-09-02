"""Feature engineering pipeline for constructing model-ready datasets."""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import sys
import yaml
import numpy as np
import pandas as pd

from src.data.schemas import (
    DATE_COLUMN,
    TARGET_COLUMNS,
    MARKET_COLUMNS,
    GPR_COLUMNS,
    WEATHER_COLUMNS,
)
from src.features.lags import create_autoregressive_lags, create_cross_vessel_lags
from src.features.transformations import (
    create_differences,
    create_percentage_changes,
    create_log_returns,
)
from src.features.rolling import (
    create_rolling_statistics,
    create_rolling_volatility,
)
from src.features.exogenous import (
    create_macro_features,
    create_geopolitical_features,
    create_weather_features,
    create_calendar_features,
)


def load_feature_config(config_path: Union[str, Path] = "configs/features.yaml") -> Dict[str, Any]:
    """Load feature engineering configuration from YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature config not found at: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_features_dataframe(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    filter_trading_days: bool = True,
) -> pd.DataFrame:
    """Build full causal feature matrix and aligned next-trading-day targets.

    Args:
        df: Clean master historical dataset.
        config: Feature configuration dictionary.
        filter_trading_days: If True, builds features on valid BDI trading sessions.

    Returns:
        pd.DataFrame: Complete feature matrix with target_{col}_next columns.
    """
    if config is None:
        config = load_feature_config()

    target_cols = config.get("targets", TARGET_COLUMNS)

    # 1. Filter to trading observations if requested
    if filter_trading_days and "is_bdi_trading_day" in df.columns:
        df_work = df[df["is_bdi_trading_day"]].copy()
    else:
        df_work = df.copy()

    df_work = df_work.sort_values(by=DATE_COLUMN, ascending=True).reset_index(drop=True)

    feature_blocks: List[pd.DataFrame] = []

    # 2. Base columns (date and current target levels as base reference)
    base_df = pd.DataFrame({
        DATE_COLUMN: df_work[DATE_COLUMN],
        "is_bdi_trading_day": df_work["is_bdi_trading_day"] if "is_bdi_trading_day" in df_work.columns else True,
    }, index=df_work.index)
    feature_blocks.append(base_df)

    # Current target levels at time t (known at end of trading day t)
    for col in target_cols:
        if col in df_work.columns:
            feature_blocks.append(pd.DataFrame({f"{col}_level": df_work[col]}, index=df_work.index))

    # 3. Autoregressive Lags
    ar_lags = config.get("lags", {}).get("autoregressive", [1, 2, 3, 5, 10, 21])
    df_ar = create_autoregressive_lags(df_work, target_cols, lags=ar_lags)
    feature_blocks.append(df_ar)

    # 4. Cross-Vessel Lags
    cross_map = config.get("lags", {}).get("cross_vessel", {
        "bdi_hsi": ["bdi_si"],
        "bdi_si": ["bdi_hsi", "bdi_pi"],
        "bdi_pi": ["bdi_si", "bdi_ci"],
        "bdi_ci": ["bdi_pi"],
    })
    cross_lags = config.get("lags", {}).get("cross_lags", [1, 5])
    df_cross = create_cross_vessel_lags(df_work, cross_map=cross_map, lags=cross_lags)
    feature_blocks.append(df_cross)

    # 5. Momentum & Change Features
    diff_windows = config.get("transformations", {}).get("diff_windows", [1, 5, 21])
    pct_windows = config.get("transformations", {}).get("pct_change_windows", [1, 5, 21])
    df_diffs = create_differences(df_work, target_cols, windows=diff_windows)
    df_pcts = create_percentage_changes(df_work, target_cols, windows=pct_windows)
    df_log_rets = create_log_returns(df_work, target_cols, windows=[1, 5])
    feature_blocks.extend([df_diffs, df_pcts, df_log_rets])

    # 6. Rolling Statistics & Volatility
    roll_windows = config.get("rolling", {}).get("windows", [7, 30])
    roll_stats = config.get("rolling", {}).get("statistics", ["mean", "std", "min", "max"])
    df_rolling = create_rolling_statistics(df_work, target_cols, windows=roll_windows, statistics=roll_stats)
    df_vol = create_rolling_volatility(df_work, target_cols, windows=roll_windows)
    feature_blocks.extend([df_rolling, df_vol])

    # 7. Exogenous Features
    exog_cfg = config.get("exogenous", {})
    energy_cols = exog_cfg.get("energy", {}).get("columns", ["wti_usd_bbl", "brent_usd_bbl"])
    fx_cols = exog_cfg.get("fx", {}).get("columns", ["usd_inr"])
    df_macro = create_macro_features(df_work, energy_cols=energy_cols, fx_cols=fx_cols, lags=[1], pct_windows=[1, 5])

    gpr_cols = exog_cfg.get("geopolitical", {}).get("columns", ["gpr", "gpr_acts", "gpr_threats"])
    df_gpr = create_geopolitical_features(df_work, gpr_cols=gpr_cols, lags=[1])

    weather_cols = exog_cfg.get("weather", {}).get("columns", ["wind_speed_max_kmh", "precip_mm", "pressure_hpa"])
    df_weather = create_weather_features(df_work, weather_cols=weather_cols, lags=[1])

    feature_blocks.extend([df_macro, df_gpr, df_weather])

    # 8. Calendar Features
    cal_feats = config.get("calendar", {}).get("features", ["month", "quarter", "day_of_week"])
    df_cal = create_calendar_features(df_work, date_col=DATE_COLUMN, features=cal_feats)
    feature_blocks.append(df_cal)

    # 9. Next-Trading-Day Targets (Shift -1 on trading observations)
    target_dict = {}
    for col in target_cols:
        if col in df_work.columns:
            # Shift -1: observation at t+1 becomes the target for features known at t
            target_dict[f"target_{col}_next"] = df_work[col].shift(-1)

    df_targets = pd.DataFrame(target_dict, index=df_work.index)

    # Combine all blocks
    full_feature_df = pd.concat(feature_blocks + [df_targets], axis=1)

    return full_feature_df


def run_feature_pipeline(config_path: Union[str, Path] = "configs/features.yaml") -> pd.DataFrame:
    """Execute end-to-end feature pipeline, validate causality, and save output dataset."""
    print("=" * 65)
    print("  FICOS - Feature Engineering Pipeline (Phase 4)")
    print("  Constructing Model-Ready Causal Feature Matrix")
    print("=" * 65)

    # 1. Load config
    config = load_feature_config(config_path)
    in_dir = Path(config.get("input", {}).get("dir", "data/processed"))
    in_filename = config.get("input", {}).get("filename", "master_dataset.csv")
    in_path = in_dir / in_filename

    out_dir = Path(config.get("output", {}).get("dir", "data/features"))
    out_filename = config.get("output", {}).get("filename", "freight_features.csv")
    out_path = out_dir / out_filename

    print(f"\n[1] Configuration:")
    print(f"    - Input Master File:   {in_path}")
    print(f"    - Output Feature File: {out_path}")

    # 2. Load master dataset
    if not in_path.exists():
        raise FileNotFoundError(f"Processed master dataset not found at: {in_path.resolve()}")

    df_master = pd.read_csv(in_path)
    df_master[DATE_COLUMN] = pd.to_datetime(df_master[DATE_COLUMN])
    print(f"\n[2] Loaded Master Dataset: {len(df_master):,} rows x {df_master.shape[1]} columns.")

    # 3. Generate feature matrix
    print(f"\n[3] Generating Causal Feature Matrix...")
    feature_df = build_features_dataframe(df_master, config=config, filter_trading_days=True)

    # 4. Sanity and Leakage checks
    print(f"\n[4] Running Integrity & Leakage Verifications...")
    # Check 1: Duplicate dates
    dup_dates = int(feature_df[DATE_COLUMN].duplicated().sum())
    if dup_dates > 0:
        raise ValueError(f"Feature dataset contains {dup_dates} duplicate dates.")

    # Check 2: Infinite values
    num_cols = feature_df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(feature_df[num_cols]).sum().sum()
    if inf_counts > 0:
        raise ValueError(f"Feature dataset contains {inf_counts} infinite values.")

    # Check 3: Date monotonicity
    if not feature_df[DATE_COLUMN].is_monotonic_increasing:
        raise ValueError("Feature dataset dates are not sorted in ascending order.")

    # Check 4: Target shift verification (t+1 alignment)
    for col in config.get("targets", TARGET_COLUMNS):
        if col in df_master.columns:
            target_col = f"target_{col}_next"
            # Target at row i should equal level at row i+1
            val_t = feature_df[f"{col}_level"].iloc[:-1].values
            target_t = feature_df[target_col].iloc[:-1].values
            level_next = feature_df[f"{col}_level"].iloc[1:].values
            if not np.allclose(target_t, level_next, equal_nan=True):
                raise ValueError(f"Target alignment mismatch for {col}!")

    print("    [PASSED] No duplicate dates.")
    print("    [PASSED] No infinite values.")
    print("    [PASSED] Monotonic chronological sorting verified.")
    print("    [PASSED] Causal target alignment (t -> t+1) mathematically verified.")

    # 5. Export features
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(out_path, index=False)
    print(f"\n[5] Export:")
    print(f"    Saved feature dataset to: {out_path.resolve()}")

    # 6. Concise Summary
    target_cols = [c for c in feature_df.columns if c.startswith("target_")]
    feat_cols = [c for c in feature_df.columns if c not in target_cols and c not in [DATE_COLUMN, "is_bdi_trading_day"]]
    date_min = feature_df[DATE_COLUMN].min().strftime("%Y-%m-%d")
    date_max = feature_df[DATE_COLUMN].max().strftime("%Y-%m-%d")

    print("\n" + "=" * 65)
    print("  FEATURE DATASET SUMMARY:")
    print("=" * 65)
    print(f"  * Source Master Rows:    {len(df_master):,}")
    print(f"  * Feature Rows (Trading):{len(feature_df):,}")
    print(f"  * Total Columns:         {feature_df.shape[1]}")
    print(f"  * Feature Variables:     {len(feat_cols)}")
    print(f"  * Target Variables:      {len(target_cols)}")
    print(f"  * Date Horizon:          {date_min} to {date_max}")
    print(f"  * Targets Available:     {len(feature_df) - 1:,} (last row is active forecast origin)")
    print(f"\n  * Feature Breakdown by Family:")
    print(f"      - Autoregressive Lags:       {len([c for c in feat_cols if '_lag_' in c and not c.startswith(('cross_', 'wti_', 'brent_', 'usd_', 'gpr_', 'wind_', 'precip_', 'pressure_'))])}")
    print(f"      - Cross-Vessel Lags:         {len([c for c in feat_cols if c.startswith('cross_')])}")
    print(f"      - Momentum / Differences:    {len([c for c in feat_cols if '_diff_' in c or '_pct_change_' in c or '_log_return_' in c])}")
    print(f"      - Rolling Stats & Volatility:{len([c for c in feat_cols if '_roll_' in c or '_return_vol_' in c])}")
    print(f"      - Macro, GPR & Weather:      {len([c for c in feat_cols if any(c.startswith(p) for p in ['wti_', 'brent_', 'usd_', 'gpr', 'wind_', 'precip_', 'pressure_'])])}")
    print(f"      - Calendar Indicators:       {len([c for c in feat_cols if c.startswith('cal_')])}")
    print("\n  STATUS: [SUCCESS] Phase 4 Feature Pipeline Execution Complete.")
    print("=" * 65 + "\n")

    return feature_df


def main():
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/features.yaml"
    try:
        run_feature_pipeline(cfg)
    except Exception as e:
        print(f"\n[FATAL ERROR] Feature pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
