"""Expanded Feature Engineering Pipeline supporting Dual Regimes (Baltic 2012-2019 & KOBC 2020+).

Builds causal features and non-interpolated next-observed-trading-day targets
from the expanded master dataset, strictly avoiding future lookahead.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import yaml
import numpy as np
import pandas as pd
import openpyxl
import xlrd

from src.data.schemas import DATE_COLUMN
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


def load_expanded_feature_config(
    config_path: Union[str, Path] = "configs/features_expanded.yaml"
) -> Dict[str, Any]:
    """Load expanded feature engineering configuration from YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found at: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_causal_monthly_commodities(
    cmo_path: Union[str, Path] = "data/raw/CMO-Historical-Data-Monthly.xlsx",
    dates_df: Optional[pd.DataFrame] = None,
    lag_months: int = 1,
) -> pd.DataFrame:
    """Causally align monthly World Bank commodity prices to daily dates.

    Publication lag: Month M prices are published after Month M ends;
    thus, on day d in Month M+1, we use Month M prices.
    """
    path = Path(cmo_path)
    if not path.exists() or dates_df is None:
        return pd.DataFrame(index=dates_df.index if dates_df is not None else None)

    try:
        wb = openpyxl.load_workbook(path, data_only=True)
        sh = wb["Monthly Prices"]
        rows = list(sh.iter_rows(values_only=True))
        header_row = rows[4]
        headers = [str(c).strip() if c is not None else f"col_{i}" for i, c in enumerate(header_row)]
        data_rows = rows[6:]
        clean_data = []
        for r in data_rows:
            if r[0] is not None and (str(r[0]).strip().startswith("19") or str(r[0]).strip().startswith("20")):
                clean_data.append(r[:len(headers)])

        df_cmo = pd.DataFrame(clean_data, columns=headers)
        
        # Parse Month
        # Format: 'YYYYMmm' e.g. '2012M08'
        cmo_records = []
        for _, row in df_cmo.iterrows():
            m_str = str(row.iloc[0]).strip()
            if "M" in m_str:
                parts = m_str.split("M")
                yr = int(parts[0])
                mo = int(parts[1])
                # Publication date is 1st of month M + lag_months
                pub_month = mo + lag_months
                pub_year = yr + (pub_month - 1) // 12
                pub_month = ((pub_month - 1) % 12) + 1
                pub_date = pd.Timestamp(year=pub_year, month=pub_month, day=1)

                def _safe_float(val):
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return np.nan

                cmo_records.append({
                    "pub_date": pub_date,
                    "cmo_coal_australian_usd": _safe_float(row.get("Coal, Australian")),
                    "cmo_coal_south_african_usd": _safe_float(row.get("Coal, South African **")),
                    "cmo_iron_ore_cfr_usd": _safe_float(row.get("Iron ore, cfr spot")),
                    "cmo_natural_gas_us_usd": _safe_float(row.get("Natural gas, US")),
                    "cmo_natural_gas_europe_usd": _safe_float(row.get("Natural gas, Europe")),
                })

        df_cmo_parsed = pd.DataFrame(cmo_records).sort_values(by="pub_date").reset_index(drop=True)
        
        # Merge as-of to daily dates
        df_target_dates = pd.DataFrame({DATE_COLUMN: pd.to_datetime(dates_df[DATE_COLUMN])})
        merged = pd.merge_asof(
            df_target_dates,
            df_cmo_parsed,
            left_on=DATE_COLUMN,
            right_on="pub_date",
            direction="backward",
        )
        drop_cols = [c for c in [DATE_COLUMN, "pub_date"] if c in merged.columns]
        merged = merged.drop(columns=drop_cols)
        return merged.set_index(dates_df.index)
    except Exception as e:
        print(f"[Warning] Failed to load monthly commodities: {e}")
        return pd.DataFrame(index=dates_df.index)


def load_causal_merchant_fleet(
    fleet_path: Union[str, Path] = "data/raw/US.MerchantFleet_20260901_183718.csv",
    dates_df: Optional[pd.DataFrame] = None,
    lag_years: int = 1,
) -> pd.DataFrame:
    """Causally align annual UNCTAD World Merchant Fleet statistics to daily dates."""
    path = Path(fleet_path)
    if not path.exists() or dates_df is None:
        return pd.DataFrame(index=dates_df.index if dates_df is not None else None)

    try:
        df_fleet = pd.read_csv(path)
        world_row = df_fleet[df_fleet["Economy_Label"] == "World"]
        if world_row.empty:
            return pd.DataFrame(index=dates_df.index)

        fleet_records = []
        for col in df_fleet.columns:
            if "Dead_weight_tons_in_thousands_Value" in col:
                yr_str = col.split("_")[0]
                if yr_str.isdigit():
                    yr = int(yr_str)
                    dwt_val = float(world_row[col].values[0])
                    # Published as of Jan 1 of yr + lag_years
                    pub_date = pd.Timestamp(year=yr + lag_years, month=1, day=1)
                    fleet_records.append({"pub_date": pub_date, "fleet_world_dwt_k": dwt_val})

        df_fleet_parsed = pd.DataFrame(fleet_records).sort_values(by="pub_date").reset_index(drop=True)
        df_target_dates = pd.DataFrame({DATE_COLUMN: pd.to_datetime(dates_df[DATE_COLUMN])})
        merged = pd.merge_asof(
            df_target_dates,
            df_fleet_parsed,
            left_on=DATE_COLUMN,
            right_on="pub_date",
            direction="backward",
        )
        drop_cols = [c for c in [DATE_COLUMN, "pub_date"] if c in merged.columns]
        merged = merged.drop(columns=drop_cols)
        return merged.set_index(dates_df.index)
    except Exception as e:
        print(f"[Warning] Failed to load merchant fleet: {e}")
        return pd.DataFrame(index=dates_df.index)


def load_causal_port_operations(
    port_path: Union[str, Path] = "data/raw/datafile.xls",
    dates_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Causally align annual major Indian port turnaround times to daily dates."""
    path = Path(port_path)
    if not path.exists() or dates_df is None:
        return pd.DataFrame(index=dates_df.index if dates_df is not None else None)

    try:
        wb = xlrd.open_workbook(path)
        sh = wb.sheet_by_index(0)
        data = [[sh.cell_value(r, c) for c in range(sh.ncols)] for r in range(sh.nrows)]
        df_port = pd.DataFrame(data[1:], columns=data[0])

        all_ports_row = df_port[df_port["Port-wise"] == "All Ports"]
        if all_ports_row.empty:
            return pd.DataFrame(index=dates_df.index)

        port_records = []
        # Columns like '2019-20', '2020-21', etc.
        for c in df_port.columns:
            parts = str(c).split("-")
            if len(parts) == 2 and parts[0].isdigit() and len(parts[0]) == 4:
                fy_start = int(parts[0])
                # FY ends March 31 of fy_start + 1. Available starting April 1 of fy_start + 1.
                pub_date = pd.Timestamp(year=fy_start + 1, month=4, day=1)
                val = float(all_ports_row[c].values[0])
                port_records.append({"pub_date": pub_date, "port_turnaround_avg_days": val})

        df_port_parsed = pd.DataFrame(port_records).sort_values(by="pub_date").reset_index(drop=True)
        df_target_dates = pd.DataFrame({DATE_COLUMN: pd.to_datetime(dates_df[DATE_COLUMN])})
        merged = pd.merge_asof(
            df_target_dates,
            df_port_parsed,
            left_on=DATE_COLUMN,
            right_on="pub_date",
            direction="backward",
        )
        drop_cols = [c for c in [DATE_COLUMN, "pub_date"] if c in merged.columns]
        merged = merged.drop(columns=drop_cols)
        return merged.set_index(dates_df.index)
    except Exception as e:
        print(f"[Warning] Failed to load port operations: {e}")
        return pd.DataFrame(index=dates_df.index)


def get_next_observed_target(
    df: pd.DataFrame,
    target_col: str,
    date_col: str = DATE_COLUMN,
    only_when_observed: bool = True,
) -> pd.Series:
    """Compute the next genuinely observed trading value strictly after each date t.

    Does NOT interpolate missing values.
    """
    valid_mask = df[target_col].notnull()
    valid_dates = df.loc[valid_mask, date_col].values
    valid_values = df.loc[valid_mask, target_col].values

    if len(valid_values) == 0:
        return pd.Series(np.nan, index=df.index, name=f"target_{target_col}_next")

    current_dates = df[date_col].values
    idx_next = np.searchsorted(valid_dates, current_dates, side="right")

    target_values = np.full(len(df), np.nan, dtype=float)
    valid_next_mask = idx_next < len(valid_values)

    if only_when_observed:
        assign_mask = valid_next_mask & valid_mask.values
    else:
        assign_mask = valid_next_mask

    target_values[assign_mask] = valid_values[idx_next[assign_mask]]

    return pd.Series(target_values, index=df.index, name=f"target_{target_col}_next")


def build_expanded_features_dataframe(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    filter_trading_days: bool = True,
) -> pd.DataFrame:
    """Build expanded causal feature matrix supporting Baltic (2012-2019) and KOBC (2020+).

    Args:
        df: Master historical dataset.
        config: Expanded feature configuration.
        filter_trading_days: If True, filters out non-trading days where neither Baltic nor KOBC is active.

    Returns:
        pd.DataFrame: Feature matrix containing causal features, regime indicators, and targets.
    """
    if config is None:
        config = load_expanded_feature_config()

    df_work = df.copy()
    df_work[DATE_COLUMN] = pd.to_datetime(df_work[DATE_COLUMN])
    df_work = df_work.sort_values(by=DATE_COLUMN, ascending=True).reset_index(drop=True)

    # 1. Determine regimes and trading flags
    baltic_targets = config.get("regimes", {}).get("baltic", {}).get("targets", ["bdi_hsi", "bdi_si", "bdi_pi", "bdi_ci"])
    kobc_targets = config.get("regimes", {}).get("kobc", {}).get("targets", ["kobc_handy", "kobc_supramax", "kobc_panamax", "kobc_cape"])
    kobc_explanatory = config.get("regimes", {}).get("kobc", {}).get("explanatory_series", ["kobc_kdci"])

    # Identify trading days per regime
    is_bdi_day = df_work[baltic_targets[0]].notnull() if (baltic_targets[0] in df_work.columns) else pd.Series(False, index=df_work.index)
    if "is_baltic_freight_day" in df_work.columns:
        is_bdi_day = is_bdi_day | df_work["is_baltic_freight_day"]

    is_kobc_day = df_work[kobc_targets[0]].notnull() if (kobc_targets[0] in df_work.columns) else pd.Series(False, index=df_work.index)
    if "is_kobc_freight_day" in df_work.columns:
        is_kobc_day = is_kobc_day | df_work["is_kobc_freight_day"]

    df_work["is_bdi_trading_day"] = is_bdi_day
    df_work["is_kobc_trading_day"] = is_kobc_day
    df_work["is_baltic_regime"] = df_work[DATE_COLUMN] < pd.Timestamp("2020-01-01")
    df_work["is_kobc_regime"] = df_work[DATE_COLUMN] >= pd.Timestamp("2020-01-01")

    # Filter trading observations if requested
    if filter_trading_days:
        active_mask = is_bdi_day | is_kobc_day
        if active_mask.any():
            df_work = df_work[active_mask].reset_index(drop=True)

    feature_blocks: List[pd.DataFrame] = []

    # 2. Base metadata block
    freight_source = []
    for dt, b_day, k_day in zip(df_work[DATE_COLUMN], df_work["is_bdi_trading_day"], df_work["is_kobc_trading_day"]):
        if k_day or dt >= pd.Timestamp("2020-01-01"):
            freight_source.append("kobc")
        elif b_day or dt < pd.Timestamp("2020-01-01"):
            freight_source.append("baltic")
        else:
            freight_source.append("none")

    base_meta = pd.DataFrame({
        DATE_COLUMN: df_work[DATE_COLUMN],
        "is_baltic_regime": df_work["is_baltic_regime"],
        "is_kobc_regime": df_work["is_kobc_regime"],
        "is_bdi_trading_day": df_work["is_bdi_trading_day"],
        "is_kobc_trading_day": df_work["is_kobc_trading_day"],
        "freight_source": freight_source,
    }, index=df_work.index)
    feature_blocks.append(base_meta)

    # 3. Freight Feature Engineering (Baltic & KOBC separately)
    ar_lags = config.get("lags", {}).get("autoregressive", [1, 2, 3, 5, 10, 21])
    diff_windows = config.get("transformations", {}).get("diff_windows", [1, 5, 21])
    pct_windows = config.get("transformations", {}).get("pct_change_windows", [1, 5, 21])
    log_windows = config.get("transformations", {}).get("log_return_windows", [1, 5])
    roll_windows = config.get("rolling", {}).get("windows", [7, 30])
    roll_stats = config.get("rolling", {}).get("statistics", ["mean", "std", "min", "max"])

    # Baltic block
    present_baltic = [c for c in baltic_targets if c in df_work.columns and df_work[c].notnull().any()]
    if present_baltic:
        for c in present_baltic:
            feature_blocks.append(pd.DataFrame({f"{c}_level": df_work[c]}, index=df_work.index))
        feature_blocks.append(create_autoregressive_lags(df_work, present_baltic, lags=ar_lags))
        baltic_cross_map = config.get("regimes", {}).get("baltic", {}).get("cross_vessel", {})
        feature_blocks.append(create_cross_vessel_lags(df_work, cross_map=baltic_cross_map, lags=[1, 5]))
        feature_blocks.append(create_differences(df_work, present_baltic, windows=diff_windows))
        feature_blocks.append(create_percentage_changes(df_work, present_baltic, windows=pct_windows))
        feature_blocks.append(create_log_returns(df_work, present_baltic, windows=log_windows))
        feature_blocks.append(create_rolling_statistics(df_work, present_baltic, windows=roll_windows, statistics=roll_stats))
        feature_blocks.append(create_rolling_volatility(df_work, present_baltic, windows=roll_windows))

    # KOBC block
    all_kobc_cols = [c for c in (kobc_targets + kobc_explanatory) if c in df_work.columns and df_work[c].notnull().any()]
    if all_kobc_cols:
        for c in all_kobc_cols:
            feature_blocks.append(pd.DataFrame({f"{c}_level": df_work[c]}, index=df_work.index))
        feature_blocks.append(create_autoregressive_lags(df_work, all_kobc_cols, lags=ar_lags))
        kobc_cross_map = config.get("regimes", {}).get("kobc", {}).get("cross_vessel", {})
        feature_blocks.append(create_cross_vessel_lags(df_work, cross_map=kobc_cross_map, lags=[1, 5]))
        feature_blocks.append(create_differences(df_work, all_kobc_cols, windows=diff_windows))
        feature_blocks.append(create_percentage_changes(df_work, all_kobc_cols, windows=pct_windows))
        feature_blocks.append(create_log_returns(df_work, all_kobc_cols, windows=log_windows))
        feature_blocks.append(create_rolling_statistics(df_work, all_kobc_cols, windows=roll_windows, statistics=roll_stats))
        feature_blocks.append(create_rolling_volatility(df_work, all_kobc_cols, windows=roll_windows))

    # 4. Daily Exogenous (Energy, FX, Geopolitical, Weather)
    exog_cfg = config.get("exogenous", {})
    energy_cols = [c for c in exog_cfg.get("energy", {}).get("columns", ["wti_usd_bbl", "brent_usd_bbl"]) if c in df_work.columns]
    fx_cols = [c for c in exog_cfg.get("fx", {}).get("columns", ["usd_inr"]) if c in df_work.columns]
    if energy_cols or fx_cols:
        feature_blocks.append(create_macro_features(df_work, energy_cols=energy_cols, fx_cols=fx_cols, lags=[1], pct_windows=[1, 5]))

    gpr_cols = [c for c in exog_cfg.get("geopolitical", {}).get("columns", ["gpr", "gpr_acts", "gpr_threats"]) if c in df_work.columns]
    if gpr_cols:
        feature_blocks.append(create_geopolitical_features(df_work, gpr_cols=gpr_cols, lags=[1]))

    weather_cols = [c for c in exog_cfg.get("weather", {}).get("columns", ["wind_speed_max_kmh", "precip_mm", "pressure_hpa"]) if c in df_work.columns]
    if weather_cols:
        feature_blocks.append(create_weather_features(df_work, weather_cols=weather_cols, lags=[1]))

    # Master dataset commodities, fleet, port turnaround
    master_comm_cols = [c for c in ["australia_coal", "south_africa_coal", "iron_ore"] if c in df_work.columns]
    if master_comm_cols:
        comm_dict = {}
        for c in master_comm_cols:
            comm_dict[f"{c}_lag_1"] = df_work[c].shift(1)
            comm_dict[f"{c}_pct_change_1"] = df_work[c].pct_change(1) * 100.0
            comm_dict[f"{c}_pct_change_21"] = df_work[c].pct_change(21) * 100.0
        feature_blocks.append(pd.DataFrame(comm_dict, index=df_work.index))

    if "world_fleet_total_dwt_thousands" in df_work.columns:
        feature_blocks.append(pd.DataFrame({
            "world_fleet_total_dwt_thousands_lag_1": df_work["world_fleet_total_dwt_thousands"].shift(1)
        }, index=df_work.index))

    port_turnaround_cols = [c for c in ["paradip_turnaround_time_days", "visakhapatnam_turnaround_time_days", "haldia_turnaround_time_days"] if c in df_work.columns]
    if port_turnaround_cols:
        port_dict = {}
        for c in port_turnaround_cols:
            port_dict[f"{c}_lag_1"] = df_work[c].shift(1)
        feature_blocks.append(pd.DataFrame(port_dict, index=df_work.index))

    # 5. Calendar Features
    cal_feats = config.get("calendar", {}).get("features", ["month", "quarter", "day_of_week"])
    feature_blocks.append(create_calendar_features(df_work, date_col=DATE_COLUMN, features=cal_feats))

    # 6. Next-Observed Targets (Shifted strictly to next genuine observation, no interpolation)
    target_series_dict = {}

    # Baltic targets
    for col in baltic_targets:
        if col in df_work.columns:
            target_series_dict[f"target_{col}_next"] = get_next_observed_target(
                df_work, col, date_col=DATE_COLUMN, only_when_observed=True
            )

    # KOBC primary vessel-class targets (KDCI is NOT a target)
    for col in kobc_targets:
        if col in df_work.columns:
            target_series_dict[f"target_{col}_next"] = get_next_observed_target(
                df_work, col, date_col=DATE_COLUMN, only_when_observed=True
            )

    df_targets = pd.DataFrame(target_series_dict, index=df_work.index)

    # Combine all feature blocks and targets
    full_df = pd.concat(feature_blocks + [df_targets], axis=1)

    return full_df


def run_expanded_feature_pipeline(
    config_path: Union[str, Path] = "configs/features_expanded.yaml"
) -> pd.DataFrame:
    """Execute the expanded feature pipeline and write to freight_features_expanded.csv."""
    cfg = load_expanded_feature_config(config_path)

    input_path = Path(cfg.get("input", {}).get("dir", "data/processed")) / cfg.get("input", {}).get("filename", "master_dataset.csv")
    output_dir = Path(cfg.get("output", {}).get("dir", "data/features"))
    output_path = output_dir / cfg.get("output", {}).get("filename", "freight_features_expanded.csv")

    if not input_path.exists():
        raise FileNotFoundError(f"Input master dataset not found at: {input_path.resolve()}")

    print(f"Loading master dataset from: {input_path}")
    df_master = pd.read_csv(input_path)
    df_master[DATE_COLUMN] = pd.to_datetime(df_master[DATE_COLUMN])

    print("Building expanded feature matrix...")
    df_features_exp = build_expanded_features_dataframe(df_master, config=cfg, filter_trading_days=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    df_features_exp.to_csv(output_path, index=False)
    print(f"Successfully saved expanded features matrix to: {output_path.resolve()}")
    print(f"Shape: {df_features_exp.shape} ({df_features_exp.shape[0]} rows x {df_features_exp.shape[1]} columns)")

    return df_features_exp


if __name__ == "__main__":
    run_expanded_feature_pipeline()
