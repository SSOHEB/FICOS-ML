# Phase 4 Feature Engineering Report

## 1. Executive Summary

This report documents the design, mathematical formulation, and data leakage verification of the Phase 4 feature engineering pipeline for the **FICOS (Freight Intelligence & Charter Optimization System)** ML component.

The pipeline transforms the clean historical master dataset (`data/processed/master_dataset.csv`) into a model-ready, strictly causal feature matrix stored in:
`data/features/freight_features.csv`

---

## 2. Forecasting Task Definition & Target Alignment

### Forecasting Horizon
The primary forecasting task is:
$$\hat{Y}_{t+1} = f(\mathbf{X}_t)$$

Where:
- $t$: Current observed Baltic Dry Index trading session (forecast origin).
- $\mathbf{X}_t$: Vector of all causal features derived strictly from historical information available at or before trading day $t$.
- $Y_{t+1}$: Next observed Baltic Dry Index trading session value for each vessel class.

### Target Alignment Formulation
For each vessel segment, the target variable is constructed by shifting forward by 1 trading session:
$$\text{target\_bdi\_hsi\_next}_t = \text{bdi\_hsi}_{t+1}$$
$$\text{target\_bdi\_si\_next}_t = \text{bdi\_si}_{t+1}$$
$$\text{target\_bdi\_pi\_next}_t = \text{bdi\_pi}_{t+1}$$
$$\text{target\_bdi\_ci\_next}_t = \text{bdi\_ci}_{t+1}$$

> [!IMPORTANT]
> **Trading-Day Alignment**: The shift operates across **consecutive trading days** rather than calendar days. Non-trading calendar days (weekends and exchange holidays) do not contaminate the target sequence or generate artificial zeros. The final trading row ($T$) has an unobserved target ($\text{NaN}$), representing the active real-world forecast origin.

---

## 3. Feature Families & Taxonomy

The pipeline generates **135 predictive features** organized into 7 functional families:

| Feature Family | Variables Generated | Parameters & Lookbacks | Rationale & Signals Captured |
| :--- | :---: | :--- | :--- |
| **1. Current Target Levels** | 4 | $t$ level reference | Base level observed at forecast origin ($I_t$). |
| **2. Autoregressive Lags** | 24 | Lags $t-1, t-2, t-3, t-5, t-10, t-21$ for each index | Captures strong temporal persistence ($ACF_1 > 0.97$) identified in Phase 3. |
| **3. Cross-Vessel Lags** | 8 | Lags $t-1, t-5$ across adjacent pairs: $\text{SI}\to\text{HSI}$, $\text{HSI}/\text{PI}\to\text{SI}$, $\text{SI}/\text{CI}\to\text{PI}$, $\text{PI}\to\text{CI}$ | Captures inter-segment charter substitution and lead-lag spillovers. |
| **4. Momentum & Differences** | 45 | Differences ($1, 5, 21$ days), % Changes ($1, 5, 21$ days), Log Returns ($1, 5$ days) | Induces stationarity, removes unit root persistence, captures market velocity. |
| **5. Rolling Stats & Volatility** | 40 | Rolling windows $W \in \{7, 30\}$: Mean, Std, Min, Max, and Return Volatility | Captures recent local price channels, support/resistance, and volatility clustering. |
| **6. Exogenous Macro & Shocks** | 24 | - Oil (WTI, Brent): Lag 1, % changes ($1, 5$)<br>- FX (USD/INR): Lag 1, % changes ($1, 5$)<br>- GPR: Lag 1, Diff 1, % change, Spike ratios ($GPR / MA30$, $GPR / MA7$)<br>- Weather: Wind speed lag 1, Pressure lag 1, Pressure diff 1, Precip lag 1 | Incorporates energy input cost pressures, currency depreciation, geopolitical conflict shocks, and regional weather disruptions. |
| **7. Calendar Indicators** | 3 | Month (1-12), Quarter (1-4), Day of Week (0-4) | Captures grain harvest season and Q4 restocking cycles. |

---

## 4. Dataset Dimensions & Summary Statistics

- **Input Master Rows**: `2,556` (full calendar series)
- **Output Feature Rows**: `1,749` (active trading sessions)
- **Total Columns**: `141`
  - Identifier & Metadata: `2` (`date`, `is_bdi_trading_day`)
  - Feature Columns: `135`
  - Target Columns: `4` (`target_bdi_hsi_next`, `target_bdi_si_next`, `target_bdi_pi_next`, `target_bdi_ci_next`)
- **Active Forecast Target Rows**: `1,748` training/evaluation instances (last row represents unobserved future step).
- **Date Range**: `2012-08-01` to `2019-07-31`.

---

## 5. Missingness & Cold-Start Analysis

| Feature Group | Expected Initial NaNs | Structural Reason |
| :--- | :---: | :--- |
| **Autoregressive Lags** | 1 to 21 rows | Maximum lag lookback ($t-21$) requires 21 historical trading sessions. |
| **Differences / % Changes** | 1 to 21 rows | Maximum difference window ($t-21$) requires 21 historical sessions. |
| **Rolling Statistics (30)** | 14 rows | Window 30 with `min_periods=15` has 14 initial cold-start rows. |
| **Target Columns** | Exactly 1 row (final row) | The very last historical observation ($T = \text{2019-07-31}$) has no $T+1$ target. |

> [!NOTE]
> **No Global Imputation**: NaNs in the early cold-start window ($t < 21$) are strictly preserved. Future forecasting models can either drop the initial 21 cold-start rows or apply model-specific causal imputation during train splits.

---

## 6. Strict Data Leakage Prevention

To guarantee zero look-ahead bias, the feature engineering pipeline enforces the following invariants:

1. **Strictly Causal Rolling Windows**: All rolling metrics specify `center=False` and operate exclusively on past historical rows:
   $$R_t = \text{Stat}(X_{t-W+1}, X_{t-W+2}, \dots, X_t)$$
2. **Causal Exogenous Alignment**: Exogenous variables (WTI, Brent, USD/INR, GPR, Weather) enter the feature matrix through causal lags ($t-1$) and backward-looking percentage changes, ensuring availability at forecast time.
3. **Future Invariance Proof**: Automated unit tests (`tests/test_features.py::test_future_independence`) artificially corrupt future data ($t \ge 20$) by a factor of $99\times$ and verify that all feature values at $t < 20$ remain bit-for-bit identical.
4. **Target Shift Verification**: Automated tests mathematically verify that $\text{target\_col}_t = \text{level\_col}_{t+1}$ across every single historical trading session.

---

## 7. Data Integrity Confirmation

- **Master Dataset Immutability**: `data/processed/master_dataset.csv` was verified using filesystem timestamps and checksums; it remains **100% bit-for-bit unmodified**.
- **Independent Feature Output**: All generated feature vectors are written exclusively to `data/features/freight_features.csv`.

---

## 8. Phase 5 Baseline Model Recommendations

When evaluating baseline and advanced forecasting models in Phase 5, the following feature subsets should be tested systematically:

1. **Pure Autoregressive Baseline (AR)**: Use only $I_t$ and autoregressive lags ($t-1, \dots, t-21$).
2. **AR + Cross-Vessel Segment Features**: Add cross-vessel lags (especially Capesize leading Panamax/Supramax).
3. **AR + Momentum & Volatility**: Add differenced returns, log returns, and rolling standard deviation channels.
4. **Full Multi-Modal Feature Set**: Add macro oil prices, FX movements, GPR shock ratios, and seasonal calendar encodings.
