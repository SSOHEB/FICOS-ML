# Phase: Rebuild Features from Expanded Master — Final Report

## 1. Executive Summary

The feature engineering pipeline was re-executed against the canonical **5,145 $\times$ 84** master dataset spanning **2012-08-01 to 2026-09-01**.

The rebuild establishes a dual-regime feature matrix:
- **Historical Baltic Exchange Regime (2012–2019):** 1,749 active trading sessions (`bdi_hsi`, `bdi_si`, `bdi_pi`, `bdi_ci`).
- **Modern KOBC Regime (2020–2026):** 1,604 active trading sessions (`kobc_handy`, `kobc_supramax`, `kobc_panamax`, `kobc_cape`, and explanatory `kobc_kdci`).

All features and targets are computed with strict mathematical causality ($t-k \le t$, `center=False`). No global interpolation or synthetic freight values were introduced. Target values reflect the next genuinely observed trading session ($t+1$).

---

## 2. Dataset Dimensions and Target Summary

| Dataset Component | Specification / Metric | Details |
| :--- | :--- | :--- |
| **Canonical Master Input Path** | `data/processed/master_dataset.csv` (and root `master_dataset.csv`) | **5,145 rows $\times$ 84 columns** |
| **Input SHA-256 Checksum** | `3930980597df212d3c3b82eb5338a2354f48b1f3d89dd61891fb5f9d11d76d54` | Ground truth canonical master |
| **Master Input Date Range** | **2012-08-01 to 2026-09-01** | 14-year complete historical timeline |
| **Expanded Features Output Path** | `data/features/freight_features_expanded.csv` | **3,353 rows $\times$ 299 columns** |
| **Output Date Range** | **2012-08-01 to 2026-09-01** | Active trading days |
| **Post-2019 KOBC Feature Rows** | **1,604 rows** (2020-01-02 to 2026-09-01) | Full post-2019 KOBC trading history |
| **Pre-2020 Baltic Feature Rows** | **1,749 rows** (2012-08-01 to 2019-07-31) | Historical Baltic trading history |
| **Engineered Features** | **285 features** | Lags, cross-lags, rolling stats, momentum, commodities, macro, GPR, weather, fleet, port turnaround, calendar |
| **Metadata & Regime Columns** | **6 columns** | `date`, `is_baltic_regime`, `is_kobc_regime`, `is_bdi_trading_day`, `is_kobc_trading_day`, `freight_source` |
| **Target Columns** | **8 columns** | 4 Baltic targets ($t+1$) + 4 KOBC targets ($t+1$) |

---

## 3. Genuine Target Observation Counts

### 3.1 KOBC Production Targets (2020+)
| Target Name | Non-Null Observations | Terminal NaN Date | Regime Isolation |
| :--- | :---: | :---: | :--- |
| `target_kobc_handy_next` | **1,603** | 2026-09-01 | Pre-2020 strictly NaN; zero Baltic contamination |
| `target_kobc_supramax_next` | **1,603** | 2026-09-01 | Pre-2020 strictly NaN; zero Baltic contamination |
| `target_kobc_panamax_next` | **1,603** | 2026-09-01 | Pre-2020 strictly NaN; zero Baltic contamination |
| `target_kobc_cape_next` | **1,603** | 2026-09-01 | Pre-2020 strictly NaN; zero Baltic contamination |
| `kobc_kdci` | **0 targets** | N/A | **Explanatory feature only** (`kobc_kdci_level`, lags, rolling stats); never a vessel target |

### 3.2 Baltic Historical Targets (2012–2019)
| Target Name | Non-Null Observations | Terminal NaN Date | Regime Isolation |
| :--- | :---: | :---: | :--- |
| `target_bdi_hsi_next` | **1,748** | 2019-07-31 | Post-2019 strictly NaN; zero KOBC contamination |
| `target_bdi_si_next` | **1,748** | 2019-07-31 | Post-2019 strictly NaN; zero KOBC contamination |
| `target_bdi_pi_next` | **1,748** | 2019-07-31 | Post-2019 strictly NaN; zero KOBC contamination |
| `target_bdi_ci_next` | **1,748** | 2019-07-31 | Post-2019 strictly NaN; zero KOBC contamination |

---

## 4. Missing-Value Summary & Rationales

| Feature / Column Group | Null Count (out of 3,353) | Rationale / Causal Behavior |
| :--- | :---: | :--- |
| `target_bdi_*_next` (Baltic targets) | 1,605 | Active only on Baltic rows (1,748 observed, 1 final Baltic NaN, 1,604 NaN during KOBC regime). |
| `target_kobc_*_next` (KOBC targets) | 1,750 | Active only on KOBC rows (1,603 observed, 1 final KOBC NaN, 1,749 NaN during Baltic regime). |
| `paradip_turnaround_time_days_lag_1` | 1,749 | Turnaround reporting series starts in FY 2019–20; unobserved during 2012–2019. |
| `usd_inr_pct_change_5`, `_1` | 196 – 203 | US/India non-overlapping bank holidays in FX series. |
| `wti_usd_bbl_pct_change_5`, `_1` | 148 – 157 | US energy market bank holidays. |
| `brent_usd_bbl_pct_change_5`, `_1` | 58 – 64 | UK/European energy market holidays. |
| Baltic / KOBC initial lags (lag_21, diff_21) | 21 per regime | Startup window for initial 21 trading sessions within each regime. |

---

## 5. Causal Leakage & Invariance Verification

- **Adversarial Future-Perturbation Test:**
  All features after cutoff $T = \text{2017-06-15}$ were corrupted by a factor of $\times 50.0 + 99,999.0$.
  Every feature at $t \le T$ demonstrated **exact bitwise/float equality ($0.000000$ deviation)**.
- **Rolling Center Validation:** Verified that rolling statistics strictly compute over $[t-k+1, t]$ with `center=False`.
- **Target Alignment Validation:** Verified targets at date $t$ reflect strictly future observed market states ($t+1$) with no interpolation.

---

## 6. Test Suite Execution & Verification

| Test Suite | Total Executed | Passed | Failed |
| :--- | :---: | :---: | :---: |
| `tests/test_expanded_features.py` | 11 | 11 | 0 |
| Core Test Suite (`test_phase10_5_*.py`, `test_forecast.py`, `test_cost.py`, `test_risk.py`, etc.) | 105 | 105 | 0 |
| **Full Combined Test Suite** | **116** | **116** | **0** |

All tests passed with zero regressions. Legacy `data/features/freight_features.csv` and production Ridge models remain completely untouched.
