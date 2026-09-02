# Phase 3 Exploratory Data Analysis & Statistical Validation Report

## 1. Executive Summary

This report presents the complete exploratory data analysis (EDA), statistical profile, and structural time-series diagnostics for the **FICOS (Freight Intelligence & Charter Optimization System)** master dataset (`data/processed/master_dataset.csv`).

The primary goals of Phase 3 are to:
1. Characterize the distributional and temporal behavior of dry-bulk freight sub-indices (**Handysize**, **Supramax**, **Panamax**, **Capesize**).
2. Quantify volatility, cross-index co-movements, and exogenous macroeconomic/geopolitical/weather associations.
3. Test stationarity and autocorrelation structures to inform Phase 4 feature engineering and subsequent model selection.

---

## 2. Dataset Overview & Calendar Breakdown

- **Source File**: `data/processed/master_dataset.csv`
- **Total Dimensions**: **2,556 rows × 20 columns**
- **Date Horizon**: **2012-08-01 to 2019-07-31** (7 continuous calendar years)
- **Temporal Frequency**: Daily calendar steps (`1D`)

### Trading vs. Non-Trading Day Breakdown
- **BDI Trading Days (`is_bdi_trading_day == True`)**: **1,749 days (68.4%)**
- **Non-Trading Days (Weekends & Baltic Holidays)**: **807 days (31.6%)**
- **Duplicate Dates**: **0** (strictly unique calendar sequence)
- **Monotonicity**: Strictly ordered in ascending chronological order (`2012-08-01` → `2019-07-31`).

---

## 3. Data Quality & Missingness Diagnostics

| Column Group | Variable(s) | Total Missing | % Missing | Trading Day Missing | Structural Explanation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Date** | `date` | 0 | 0.0% | 0 | Complete index |
| **Freight Targets** | `bdi_hsi`, `bdi_si`, `bdi_pi`, `bdi_ci` | 807 | 31.57% | **0** | Missingness strictly corresponds to weekends & Baltic Exchange market closures. |
| **Energy** | `wti_usd_bbl` | 798 | 31.22% | 39 | Missingness corresponds to US holidays (Thanksgiving, Memorial Day, etc.). |
| **Energy** | `brent_usd_bbl` | 777 | 30.40% | 14 | Missingness corresponds to European market holidays. |
| **Exchange Rate** | `usd_inr` | 806 | 31.53% | 54 | Missingness corresponds to Federal Reserve bank holidays. |
| **Geopolitical Risk** | `gpr_*` (6 variables) | 0 | 0.0% | 0 | Continuous daily media text index. |
| **Weather** | `wind_*`, `precip_mm`, `pressure_hpa` | 0 | 0.0% | 0 | Continuous daily meteorological records. |
| **Trading Flags** | `is_bdi_trading_day`, `is_market_trading_day` | 0 | 0.0% | 0 | Fully populated binary flags. |

> [!NOTE]
> **Zero Target Anomalies**: There are 0 missing observations across any of the 4 freight targets on active trading days. Missing values reflect real-world exchange closures and are preserved as `NaN` without synthetic imputation.

---

## 4. Freight Target Statistical Profiles

The four target series reflect distinct vessel sizes (DWT capacity) in the dry bulk sector:
1. **Handysize (`bdi_hsi`)**: Smallest dry bulk vessels (15k - 35k DWT)
2. **Supramax (`bdi_si`)**: Medium-small dry bulk vessels (50k - 60k DWT)
3. **Panamax (`bdi_pi`)**: Medium-large dry bulk vessels (65k - 80k DWT)
4. **Capesize (`bdi_ci`)**: Largest vessels (100k - 200k+ DWT, primarily iron ore and coal)

### Target Summary Statistics (1,749 Trading Days)

| Target Variable | Mean | Median | Std Dev | CV (%) | Min | 25th % | 75th % | 90th % | Max | Skewness | Kurtosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`bdi_hsi`** (Handysize) | 480.94 | 475.00 | 119.37 | 24.82% | 183.00 | 399.00 | 561.00 | 638.00 | 821.00 | 0.08 | -0.27 |
| **`bdi_si`** (Supramax) | 832.87 | 845.00 | 213.67 | 25.65% | 243.00 | 688.00 | 959.00 | 1,089.20 | 1,562.00 | 0.15 | 0.60 |
| **`bdi_pi`** (Panamax) | 1,030.34 | 992.00 | 388.36 | 37.69% | 282.00 | 723.00 | 1,319.00 | 1,553.00 | 2,219.00 | 0.41 | -0.39 |
| **`bdi_ci`** (Capesize) | 1,704.38 | 1,587.00 | 905.33 | **53.12%** | 92.00 | 1,056.00 | 2,245.00 | 3,038.60 | 4,438.00 | 0.59 | 0.00 |

### Key Observations:
1. **Volatile Vessel Hierarchy**: Dispersion and Coefficient of Variation scale systematically with vessel size:
   - Handysize CV: **24.8%**
   - Supramax CV: **25.7%**
   - Panamax CV: **37.7%**
   - Capesize CV: **53.1%**
2. **Capesize Extreme Dynamic Range**: Capesize moves across an extreme range from a historical trough of 92 to a peak of 4,438 (a ~48x expansion factor).

---

## 5. Cross-Target Relationships & Correlation

### Cross-Index Correlation Matrix
| Metric | `bdi_hsi` | `bdi_si` | `bdi_pi` | `bdi_ci` |
| :--- | :---: | :---: | :---: | :---: |
| **`bdi_hsi`** | **1.000** | 0.893 | 0.724 | 0.601 |
| **`bdi_si`** | 0.893 | **1.000** | 0.867 | 0.751 |
| **`bdi_pi`** | 0.724 | 0.867 | **1.000** | 0.849 |
| **`bdi_ci`** | 0.601 | 0.751 | 0.849 | **1.000** |

### Insights:
- Adjacent vessel segments exhibit the strongest linear coupling (e.g. HSI & SI: $r = 0.893$; PI & CI: $r = 0.849$).
- Cross-market substitution effects exist between neighboring vessel sizes.
- Capesize and Handysize have the lowest mutual coupling ($r = 0.601$), consistent with differing cargo types (Capesize is pure major bulk / iron ore, whereas Handysize handles minor bulks, grain parcels, and steel products).

---

## 6. External Variable Analysis

### Correlation with Exogenous Drivers
| Driver Variable | `bdi_hsi` | `bdi_si` | `bdi_pi` | `bdi_ci` | Notes / Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`wti_usd_bbl`** (Crude Oil) | -0.198 | -0.165 | -0.103 | +0.075 | Complex multi-cycle co-movement across the 2014-2016 commodity collapse |
| **`brent_usd_bbl`** (Crude Oil) | -0.187 | -0.158 | -0.102 | +0.069 | Highly collinear with WTI ($r = 0.985$) |
| **`usd_inr`** (FX Rate) | -0.428 | -0.342 | -0.126 | +0.045 | Moderately negative linear association with smaller vessel rates |
| **`gpr`** (Geopolitical Risk) | +0.141 | +0.128 | +0.089 | +0.038 | Mild positive contemporaneous association |
| **`wind_speed_max_kmh`** | -0.062 | -0.055 | -0.041 | -0.025 | Local weather exhibits low contemporaneous linear correlation with global freight indices |
| **`pressure_hpa`** | +0.088 | +0.072 | +0.061 | +0.048 | Seasonal barometric variation aligns mildly with winter shipping cycles |

---

## 7. Volatility & Return Dynamics

Analyzing day-to-day percentage changes on trading days:

| Freight Target | Daily Mean Return | Daily Std (%) | Min Daily Change | Max Daily Jump | Annualized Volatility |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`bdi_hsi`** | -0.00% | 1.05% | -6.37% | +5.47% | **16.62%** |
| **`bdi_si`** | +0.01% | 1.31% | -8.47% | +10.04% | **20.75%** |
| **`bdi_pi`** | +0.07% | 2.47% | -11.29% | +14.90% | **39.14%** |
| **`bdi_ci`** | +0.26% | 6.59% | -27.82% | **+101.92%** | **104.55%** |

### Insights:
- **Fat Tails & Skewness**: Capesize exhibits massive daily jumps (up to +101.9% in a single trading session during sudden chartering squeezes).
- **Volatility Clustering**: Large freight rate movements are grouped in distinct market regimes (e.g. Q3 2013, Q4 2014, and Q3 2019).

---

## 8. Stationarity Analysis

Statistical unit root and stationarity tests were conducted on active trading observations:

| Target | ADF Statistic | ADF p-value | ADF Conclusion (5%) | KPSS Statistic | KPSS p-value | KPSS Conclusion (5%) | Joint Interpretation |
| :--- | :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| **`bdi_hsi`** | -2.4971 | 0.1162 | **Non-Stationary** (Fail to reject $H_0$) | 0.6125 | 0.0215 | **Non-Stationary** (Reject $H_0$) | **I(1) Integrated Series** |
| **`bdi_si`** | -3.2397 | 0.0178 | Stationary (Reject $H_0$) | 0.5511 | 0.0302 | **Non-Stationary** (Reject $H_0$) | **Trend / Cycle Persistence** |
| **`bdi_pi`** | -3.8359 | 0.0026 | Stationary (Reject $H_0$) | 1.3546 | 0.0100 | **Non-Stationary** (Reject $H_0$) | **Trend / Cycle Persistence** |
| **`bdi_ci`** | -4.5645 | 0.0002 | Stationary (Reject $H_0$) | 0.3503 | 0.0986 | Stationary (Fail to reject $H_0$) | **Mean-Reverting Extremes** |

### Modeling Takeaways:
- Long-memory levels require careful differencing, percentage-change transformations, or autoregressive structures during feature engineering.

---

## 9. Temporal Structure & Autocorrelation

| Target | Lag 1 ACF | Lag 5 ACF | Lag 10 ACF | Lag 20 ACF | Lag 30 ACF |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`bdi_hsi`** | 0.998 | 0.985 | 0.963 | 0.908 | 0.845 |
| **`bdi_si`** | 0.997 | 0.978 | 0.947 | 0.871 | 0.785 |
| **`bdi_pi`** | 0.990 | 0.923 | 0.840 | 0.697 | 0.579 |
| **`bdi_ci`** | 0.978 | 0.865 | 0.737 | 0.536 | 0.407 |

- **Very High Autoregressive Persistence**: Lag-1 ACF exceeds 0.97 across all 4 series.
- **Autoregressive Memory**: Past freight levels and lagged returns will form primary baseline predictors for forecasting models.

---

## 10. Outlier Diagnostics

Using IQR $1.5\times$ threshold on trading observations:
- **`bdi_hsi`**: 8 outliers (0.46%) in range [805, 821] during the late 2013 freight rally.
- **`bdi_si`**: 37 outliers (2.12%) during the early 2016 shipping trough ([243, 281]) and late 2013 peak ([1366, 1562]).
- **`bdi_pi`**: 1 outlier (0.06%) at 2,219 in July 2019.
- **`bdi_ci`**: 23 outliers (1.32%) exceeding 4,028 during chartering squeezes in Dec 2013, Dec 2017, and July 2019.

> [!IMPORTANT]
> **Conclusion on Outliers**: All identified outliers correspond to genuine historical market rallies and shipping downturns (e.g. 2016 market crash and 2019 Vale dam collapse aftermath). None are data-entry errors; they must be retained in the dataset.

---

## 11. Recommendations for Phase 4 (Feature Engineering)

Based on statistical validation, the following feature categories should be investigated in Phase 4:

1. **Autoregressive Freight Lags**:
   - Short-term lags: $t-1, t-2, t-3, t-5$ (1 week).
   - Medium-term lags: $t-10, t-21$ (1 month trading horizon).
2. **Cross-Index Lead-Lag Interactions**:
   - Capesize leading Panamax or Supramax co-movement indicators.
3. **Rolling Volatility & Momentum**:
   - Rolling standard deviations (7-day, 30-day windows).
   - Rolling exponential moving averages (EMA-7, EMA-30).
4. **Rate of Change / Return Signals**:
   - 1-day, 5-day, and 21-day log differences / percentage changes.
5. **Macroeconomic & Exogenous Lags**:
   - Lagged oil price returns (`wti_usd_bbl`, `brent_usd_bbl`).
   - Geopolitical risk momentum and spikes (`gpr_threats`, `gpr_ma7`).
   - Exchange rate shifts (`usd_inr`).
6. **Calendar & Seasonality Features**:
   - Month-of-year and quarter indicators to capture grain harvest and winter restocking seasonality.
