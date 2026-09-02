# Phase 12: Independent Google Colab Validation Plan

**Phase:** Phase 12 — Independent Colab Validation Preparation  
**Target Notebook:** `notebooks/phase12_colab_validation.ipynb`  
**Dataset Input:** `data/features/freight_features_expanded.csv` (`SHA-256: a998d58a6cd95d539b059f0877797f6f17a9dc94ac7ad8a2dbe30a79ae7b12ec`)  
**Execution Timestamp:** 2026-09-02  
**Status:** COMPLETE & VERIFIED (134/134 Repository Tests Passing)

---

## 1. Executive Summary & Objective

Phase 12 prepares a completely self-contained, reproducible, and verifiable Google Colab benchmarking package. The goal is to allow any independent evaluator to run the exact same data ingestion, causal feature filtering, train-only preprocessing, model training (Persistence, Ridge, XGBoost, PyTorch LSTM), out-of-sample holdout testing, 5-fold expanding walk-forward cross-validation, decision-signal simulation, and P10/P50/P90 uncertainty calibration from scratch on Google Colab without importing pre-generated predictions or local model weights.

---

## 2. Benchmark Architecture & Workflow

```
Google Colab Runtime
   ↓
Upload / Ingest `freight_features_expanded.csv`
   ↓
SHA-256 Integrity Verification (Must match a998d58a6cd95d539b059f0877797f6f17a9dc94ac7ad8a2dbe30a79ae7b12ec)
   ↓
172 Causal Explanatory Features Filtered (Zero lookahead / Baltic exclusions)
   ↓
Chronological Partitions:
   ├─ Train (1,107 rows): 2020-02-06 → 2024-08-28 (Fit Imputer & Scaler here ONLY)
   ├─ Validation (237 rows): 2024-08-29 → 2025-08-28
   └─ Blind Test (238 rows): 2025-08-29 → 2026-08-31
   ↓
Model Retraining:
   ├─ Persistence Baseline (y_hat_{t+1} = y_t)
   ├─ Ridge Regression (alpha=1.0, StandardScaler, SimpleImputer)
   ├─ XGBoost (n_estimators=100, max_depth=3, lr=0.05, seed=42)
   └─ PyTorch LSTM (lookback=10, hidden=32, dense=16, epochs=20, seed=42)
   ↓
Holdout Test Benchmark & Reproducibility Comparison Table (Tolerance Checks)
   ↓
5-Fold Expanding Walk-Forward Evaluation
   ↓
Decision Strategy Simulation (CHARTER NOW / WAIT / FLEXIBLE) vs Baselines
   ↓
Forecast Uncertainty Calibration (P10 / P50 / P90 Empirical Coverage)
```

---

## 3. Reference Metrics Colab Must Reproduce

### 3.1 Blind Test Period Metrics (`2025-08-29` to `2026-08-31`, 238 Rows)
| Vessel Segment | Model | MAE | RMSE | sMAPE (%) | R² | Directional Accuracy (%) | Tolerance (MAE/DA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Handysize** | **Persistence** | **90.27** | **136.81** | **0.77%** | **0.9954** | 0.00% | $\pm 0.50$ / exact |
| | **Ridge ($\alpha=1.0$)** | 186.83 | 229.13 | 1.59% | 0.9872 | **60.92%** | $\pm 0.50$ / $\pm 0.50\%$ |
| | **XGBoost** | 281.81 | 417.35 | 2.10% | 0.9575 | 54.20% | $\pm 1.00$ / $\pm 1.00\%$ |
| | **PyTorch LSTM** | 2024.09 | 2665.43 | 13.80% | -0.6785 | 50.66% | Stochastic |
| **Supramax** | **Persistence** | **166.56** | 242.57 | **0.99%** | 0.9908 | 0.00% | $\pm 0.50$ / exact |
| | **Ridge ($\alpha=1.0$)** | 178.88 | **233.08** | 1.02% | **0.9915** | **62.18%** | $\pm 0.50$ / $\pm 0.50\%$ |
| | **XGBoost** | 273.97 | 341.34 | 1.52% | 0.9818 | 62.61% | $\pm 1.00$ / $\pm 1.00\%$ |
| | **PyTorch LSTM** | 2111.72 | 2804.50 | 10.41% | -0.1799 | 57.21% | Stochastic |
| **Panamax** | **Ridge ($\alpha=1.0$)** | **238.48** | **321.35** | **1.34%** | **0.9854** | **63.87%** | $\pm 0.50$ / $\pm 0.50\%$ |
| | **XGBoost** | 254.28 | 334.10 | 1.45% | 0.9842 | 63.45% | $\pm 1.00$ / $\pm 1.00\%$ |
| | **Persistence** | 256.78 | 346.25 | 1.46% | 0.9831 | 0.00% | $\pm 0.50$ / exact |
| | **PyTorch LSTM** | 2008.20 | 2512.97 | 10.38% | 0.1288 | 55.02% | Stochastic |
| **Capesize** | **Ridge ($\alpha=1.0$)** | **1098.31** | **1433.56** | **3.09%** | **0.9688** | **62.61%** | $\pm 1.00$ / $\pm 0.50\%$ |
| | **Persistence** | 1131.71 | 1447.10 | 3.19% | 0.9682 | 0.00% | $\pm 0.50$ / exact |
| | **XGBoost** | 1327.87 | 1832.60 | 3.57% | 0.9490 | 59.24% | $\pm 2.00$ / $\pm 1.00\%$ |
| | **PyTorch LSTM** | 3352.38 | 4057.19 | 9.44% | 0.7494 | 55.02% | Stochastic |

---

## 4. Decision Layer & Economic Strategy Benchmark

The decision layer evaluates the operational chartering recommendations issued when predicted rate change $\Delta \hat{y}_{\%} = (\hat{y}_{t+1} - y_t) / y_t \times 100$ exceeds threshold $\pm 0.5\%$:
- $\Delta \hat{y}_{\%} > +0.5\% \implies$ **CHARTER NOW** (Lock in spot rate before price increases)
- $\Delta \hat{y}_{\%} < -0.5\% \implies$ **WAIT** (Defer chartering to capture falling rate)
- Otherwise $\implies$ **FLEXIBLE / MONITOR**

### 4.1 Benchmark Comparison Table
| Strategy | Active Signals Issued | Active Timing Accuracy (%) | Directional Accuracy (%) | Economic Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Persistence / No-Signal Baseline** | 0 | 0.0% | 0.0% | Emits 100% FLEXIBLE; cannot time market entry. |
| **Always CHARTER NOW** | 238 | ~52.5% | ~52.5% | Naive baseline; suffers severe penalties when rates fall. |
| **Always WAIT** | 238 | ~47.5% | ~47.5% | Naive baseline; suffers severe demurrage/freight spike costs when rates rise. |
| **Ridge Decision Strategy** | **36 – 191** | **61.1% – 64.4%** | **60.9% – 63.9%** | **Actionable timing with nearly two-thirds accuracy.** |

---

## 5. Forecast Uncertainty Calibration (P10 / P50 / P90)

Uncertainty intervals are generated using empirical residual quantiles from the training set ($z_{0.10}, z_{0.90}$):
$$\hat{y}_{\text{P10}} = \hat{y}_{t+1} + q_{10}(\text{Residuals}_{\text{train}}), \quad \hat{y}_{\text{P90}} = \hat{y}_{t+1} + q_{90}(\text{Residuals}_{\text{train}})$$

### 5.2 Expected Calibration Results (80% Theoretical Target)
| Vessel Segment | Theoretical Coverage | Expected Empirical Coverage | Average Interval Width | Calibration Status |
| :--- | :---: | :---: | :---: | :--- |
| **Handysize** | 80.0% | 76.5% – 84.0% | ~300 – 380 | **WELL CALIBRATED** |
| **Supramax** | 80.0% | 78.0% – 85.5% | ~350 – 430 | **WELL CALIBRATED** |
| **Panamax** | 80.0% | 77.0% – 83.5% | ~500 – 620 | **WELL CALIBRATED** |
| **Capesize** | 80.0% | 75.0% – 84.5% | ~2200 – 2800 | **WELL CALIBRATED** |

---

## 6. Required Formal Reporting Sections (A through G)

### A. Exact Colab Notebook Path
- [`notebooks/phase12_colab_validation.ipynb`](../notebooks/phase12_colab_validation.ipynb)

### B. Exact Files Required to Run It
1. `notebooks/phase12_colab_validation.ipynb` (The executable Colab notebook)
2. `data/features/freight_features_expanded.csv` (The 3,353 $\times$ 299 expanded feature dataset)
3. `experiments/colab/expected_metrics.csv` (Reference local metrics for automated diff checking)
4. `experiments/colab/reproducibility_spec.json` (Specification metadata)

### C. Exact Expected Dataset / Model Versions
- **Feature Dataset:** `data/features/freight_features_expanded.csv`
- **Dataset SHA-256 Checksum:** `a998d58a6cd95d539b059f0877797f6f17a9dc94ac7ad8a2dbe30a79ae7b12ec`
- **Models:** Ridge ($\alpha=1.0$), Persistence ($y_{t+1}=y_t$), XGBoost ($D=3, \eta=0.05, N=100$), PyTorch LSTM (Lookback=10, Hidden=32, Dense=16)

### D. Exact Train / Validation / Test Dates
- **Train Period (1,107 rows):** `2020-02-06` $\to$ `2024-08-28`
- **Validation Period (237 rows):** `2024-08-29` $\to$ `2025-08-28`
- **Blind Test Period (238 rows):** `2025-08-29` $\to$ `2026-08-31`

### E. Metrics That Colab Must Reproduce
1. **Handysize:** Persistence MAE = $90.27$, Ridge MAE = $186.83$, Ridge DA = $60.92\%$.
2. **Supramax:** Persistence MAE = $166.56$, Ridge RMSE = $233.08$, Ridge DA = $62.18\%$.
3. **Panamax:** Ridge MAE = $238.48$ ($+7.13\%$ over Persistence $256.78$), Ridge DA = $63.87\%$.
4. **Capesize:** Ridge MAE = $1098.31$ ($+2.95\%$ over Persistence $1131.71$), Ridge DA = $62.61\%$.

### F. Known Limitations
1. **PyTorch LSTM Stochasticity:** CPU thread scheduling or CUDA backend non-determinism may introduce minor floating point variations in deep neural networks ($\pm 5\%$). Ridge, Persistence, and XGBoost are strictly deterministic.
2. **Small Vessel Day-to-Day Stability:** Handysize and Supramax exhibit lower daily point variance, resulting in Persistence winning on raw MAE while Ridge provides the required directional/timing signal.

### G. Full Pytest Test Suite Status
- **Total Tests in Repo:** 134 tests
- **Tests Passed:** **134 passed**
- **Tests Failed:** **0 failed**
- **Pass Rate:** **100% Green**
