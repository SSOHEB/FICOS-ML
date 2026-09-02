# Phase 2 Data Audit & Ingestion Report

## 1. Executive Summary

This report documents the Phase 2 data pipeline establishment for the **FICOS (Freight Intelligence & Charter Optimization System)** ML component. The pipeline loads, validates, normalizes, and exports the clean historical master dataset into `data/processed/master_dataset.csv` without artificial data creation, forward-filling, or time-series leakage.

---

## 2. Datasets Discovered & Audit Summary

| Dataset File | Stated Source / Scope | Frequency | Date Range | Decision | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `master_daily.csv` | Unified Freight, Commodity, Macro, GPR & Weather Series | Daily | 2012-08-01 to 2019-07-31 | **INCLUDED** | Curated daily master series containing dry-bulk Baltic indices, oil prices, FX, GPR, and weather variables. |
| `edited BDI data1.xls` | Baltic Dry Index Historical Sub-indices | Daily (Trading) | 2012-08-01 to 2019-07-31 | Subsumed | Exact trading-day source underlying `bdi_*` in `master_daily.csv`. |
| `PET_PRI_SPT_S1_D.xls` | EIA Cushing WTI & Europe Brent Spot Prices | Daily (Trading) | 1986-01-02 to present | Subsumed | Underlying daily crude oil source present in `master_daily.csv`. |
| `DEXINUS.csv` | FRED USD/INR Foreign Exchange Rate | Daily (Trading) | 1973-01-02 to present | Subsumed | Underlying FX rate series present in `master_daily.csv`. |
| `data_gpr_daily_recent.xls` | Caldara & Iacoviello Geopolitical Risk Index | Daily | 1985-01-01 to present | Subsumed | Underlying daily GPR indices and article counts present in `master_daily.csv`. |
| `open-meteo-20.35N86.65E4m.csv` | Coastal Weather Observations (Paradeep Port Area) | Daily | 2010-01-01 to present | Subsumed | Underlying daily wind, precipitation, and pressure series present in `master_daily.csv`. |
| `CMO-Historical-Data-Monthly.xlsx` | World Bank Commodity Markets Outlook Pink Sheet | Monthly | 1960-01 to 2026-07 | Excluded (Phase 2) | Monthly frequency; omitted from core daily time-series to prevent low-frequency misalignment in Phase 2. |
| `gscpi_data.xls` | NY Fed Global Supply Chain Pressure Index | Monthly | 1997-09 to 2026-07 | Excluded (Phase 2) | Monthly frequency; omitted from core daily master forecasting table. |
| `datafile.xls` | Indian Major Ports Cargo Traffic (Ministry of Shipping) | Annual | 2019-20 to 2023-24 | Excluded (Phase 2) | Highly aggregated annual port summaries; unsuitable for high-frequency time-series forecasting. |
| `US.MerchantFleet_20260901_183718.csv` | UNCTAD World Merchant Fleet Statistics | Annual | 1980 to 2026 | Excluded (Phase 2) | Annual summary data; not aligned with daily forecasting horizon. |

---

## 3. Master Dataset Schema & Variables

The master dataset consists of **20 variables** organized into functional categories:

```text
1. Date Identifier:
   - date: Calendar date (YYYY-MM-DD, UTC normalized)

2. Forecasting Targets (Baltic Dry Sub-Indices):
   - bdi_hsi: Baltic Handysize Index
   - bdi_si:  Baltic Supramax Index
   - bdi_pi:  Baltic Panamax Index
   - bdi_ci:  Baltic Capesize Index

3. Energy & Foreign Exchange:
   - wti_usd_bbl:   WTI Crude Spot Price (USD/bbl)
   - brent_usd_bbl: Europe Brent Spot Price (USD/bbl)
   - usd_inr:       USD to Indian Rupee Exchange Rate

4. Geopolitical Risk (GPR):
   - gpr:               Geopolitical Risk Daily Index
   - gpr_acts:          GPR Acts component
   - gpr_threats:       GPR Threats component
   - gpr_ma7:           7-day moving average of GPR
   - gpr_ma30:          30-day moving average of GPR
   - gpr_article_count: Daily newspaper article volume count

5. Weather Conditions (Shipping Corridor / Port):
   - wind_speed_max_kmh: Daily maximum wind speed (km/h)
   - wind_gust_max_kmh:  Daily maximum wind gust (km/h)
   - precip_mm:          Daily total precipitation (mm)
   - pressure_hpa:       Daily mean surface atmospheric pressure (hPa)

6. Calendar / Market Flags:
   - is_bdi_trading_day:    Boolean flag (True if Baltic Exchange traded)
   - is_market_trading_day: Boolean flag (True if broader energy/FX traded)
```

---

## 4. Quality, Missing Values, and Time-Series Integrity

### Calendar Alignment & Frequency
- **Frequency**: Daily calendar dates (`1D` step).
- **Date Range**: `2012-08-01` to `2019-07-31` (exactly **2,556 calendar days** / 7.0 continuous years).
- **Duplicate Dates**: **0** (strictly unique daily index).
- **Monotonicity**: Strictly sorted chronologically in ascending order.

### Missing Value Analysis
- Total Records: **2,556**
- Trading Days: **1,749** days (68.4%)
- Non-Trading Days (Weekends & UK/US Holidays): **807** days (31.6%)

| Column | Missing Count | Missing % | Semantic Meaning |
| :--- | :---: | :---: | :--- |
| `bdi_hsi` | 807 | 31.6% | Non-trading days for Baltic Handysize (weekends & exchange holidays) |
| `bdi_si` | 807 | 31.6% | Non-trading days for Baltic Supramax |
| `bdi_pi` | 807 | 31.6% | Non-trading days for Baltic Panamax |
| `bdi_ci` | 807 | 31.6% | Non-trading days for Baltic Capesize |
| `wti_usd_bbl` | 798 | 31.2% | US Energy market holidays & weekends |
| `brent_usd_bbl` | 777 | 30.4% | European Energy market holidays & weekends |
| `usd_inr` | 806 | 31.5% | Federal Reserve FX non-trading days & weekends |
| `gpr_*` (all 6) | 0 | 0.0% | Continuous calendar daily observations |
| `weather_*` (all 4) | 0 | 0.0% | Continuous weather station daily observations |
| `is_*_trading_day` | 0 | 0.0% | Fully specified boolean flags |

### Strict Integrity Rules Enforced
1. **No Target Imputation / Interpolation**: Baltic index targets are intentionally kept as `NaN` on non-trading days. They are never artificially filled to prevent look-ahead bias or synthetic volatility smoothing.
2. **No Random Shuffling**: The series maintains strict temporal ordering.
3. **No Premature Feature Engineering**: Lags, rolling metrics, and differences will be engineered during Phase 4.

---

## 5. Output Verification

- **Output File**: `data/processed/master_dataset.csv`
- **Output Shape**: `(2556, 20)`
- **Determinism**: The pipeline is fully deterministic and reproducible.
