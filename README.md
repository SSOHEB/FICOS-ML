# FICOS — Freight Intelligence & Charter Optimization System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Pytest](https://img.shields.io/badge/pytest-134%20passed%20(100%25)-brightgreen.svg)](tests/)
[![Dataset SHA-256](https://img.shields.io/badge/dataset%20SHA--256-verified-success.svg)](data/features/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**FICOS** (Freight Intelligence & Charter Optimization System) is an enterprise-grade quantitative machine learning and decision-support engine developed for **Smart India Hackathon (SIH26006)**. It forecasts dry-bulk ocean freight rate dynamics across 4 standard vessel classes (**Handysize, Supramax, Panamax, Capesize**) and translates econometric forecasts into risk-managed, cost-optimal chartering decisions for dry-bulk commodity charterers and port operators.

---

## 1. Problem Statement & Operational Context (SIH26006)

International dry-bulk freight rates exhibit extreme volatility driven by commodity demand, bunker fuel prices, currency fluctuations, port congestion, and geopolitical disruptions. Chartering decisions (spot chartering vs. forward contracts) are traditionally made using ad-hoc intuition, exposing shipping operations to severe freight price spikes and demurrage penalties.

**FICOS solves this by providing:**
1. **Multi-Horizon Freight Forecasting:** Rigorous point and probabilistic forecasts ($t+1$ to multi-step).
2. **Causal Exogenous Integration:** Unifies macroeconomic series (WTI/Brent crude, coal, iron ore), foreign exchange (USD/INR), and Geopolitical Risk (GPR) indices with strict zero-lookahead temporal alignment.
3. **Operational Decision Optimization:** Computes vessel suitability, port draft constraints, voyage-level cost equations, demurrage/fuel exposure, and issues concrete timing signals (**`CHARTER NOW`**, **`WAIT`**, or **`FLEXIBLE / MONITOR`**).

---

## 2. System Architecture

```
                                  [ Historical Ingestion ]
                 Baltic (2012–2019) & KOBC (2020–2026) Canonical Series
                                             │
                                             ▼
                             [ Data Validation & Integrity ]
                             Checksums, Ordering, Imputation
                                             │
                                             ▼
                        [ 172-Feature Causal Feature Engine ]
            Autoregressive Lags, Rolling Extrema, Cross-Vessel Dynamics,
                  KDCI Composite, Commodities, Crude, FX, GPR
                                             │
                                             ▼
                               [ Forecasting Model Suite ]
      ┌─────────────────────┬─────────────────────┬─────────────────────┐
      │   Persistence (t)   │  Ridge (alpha=1.0)  │  XGBoost Regressor  │  PyTorch LSTM
      └─────────────────────┴─────────────────────┴─────────────────────┘
                                             │
                                             ▼
                          [ P10 / P50 / P90 Uncertainty Engine ]
                              Empirical Residual Quantiles
                                             │
                                             ▼
                            [ Decision & Optimization Layer ]
               Vessel Suitability ── Port Draft Bounds ── Total Cost Model
                                             │
                                             ▼
                            [ Structured Charter Recommendation ]
                 Action: CHARTER NOW / WAIT / FLEXIBLE | Risk Classification
```

---

## 3. Data Pipeline & Market Regimes

The modeling dataset strictly isolates two non-interchangeable maritime freight eras:
- **Baltic Era (2012–2019):** Baltic Dry Index segments (`bdi_hsi`, `bdi_si`, `bdi_pi`, `bdi_ci`).
- **KOBC Era (2020–2026):** Korea Ocean Business Corporation dry bulk indices (`kobc_handy`, `kobc_supramax`, `kobc_panamax`, `kobc_cape`, and explanatory `kobc_kdci`).

### Chronological Partitions (Post-2020 KOBC Production Dataset)
- **Train (70%, 1,107 rows):** `2020-02-06` to `2024-08-28` *(All scalers & imputers fit here only)*
- **Validation (15%, 237 rows):** `2024-08-29` to `2025-08-28`
- **Blind Test (15%, 238 rows):** `2025-08-29` to `2026-08-31` *(Out-of-sample holdout)*

---

## 4. Empirical Benchmark & Model Selection (Phase 11 Audit)

Evaluated on the 238 blind test trading sessions (`2025-08-29` to `2026-08-31`):

| Vessel Class | Best Point Model | Point MAE | Best Timing Model | Directional Accuracy | Actionable Signal Accuracy | Recommended Strategy |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| **Handysize** | **Persistence** | **90.27** | **Ridge** | **60.92%** | **61.11%** | **Hybrid:** Persistence for base point forecast + Ridge for market-entry timing. |
| **Supramax** | **Persistence** | **166.56** | **Ridge** | **62.18%** | **62.03%** | **Hybrid:** Persistence point anchor + Ridge risk/timing signals (Ridge RMSE 233.08). |
| **Panamax** | **Ridge ($\alpha=1.0$)** | **238.48** | **Ridge** | **63.87%** | **64.42%** | **Ridge Forecaster:** Strictly outperforms Persistence across MAE, RMSE, sMAPE, and DA. |
| **Capesize** | **Ridge ($\alpha=1.0$)** | **1098.31** | **Ridge** | **62.61%** | **64.40%** | **Ridge Forecaster:** Outperforms Persistence across all metrics (+13.15% error reduction in falling markets). |

> **Critical Note on Model Honesty:** Ridge does **not** universally beat Persistence on raw point error across all vessel classes. On smaller vessels with narrow daily rate steps (Handysize/Supramax), Persistence achieves lower MAE because it never deviates from $y_t$. However, Persistence emits **0 active timing signals** (0.0% DA). Ridge provides the critical directional signal ($\text{DA} > 60\%$) required for timing freight lock-in.

---

## 5. Independent Google Colab Validation (Phase 12)

A standalone, self-contained Google Colab validation notebook is provided to independently reproduce all data processing, model training, cross-validation, and decision evaluation without relying on pre-computed local artifacts.

- **Colab Notebook:** [`notebooks/phase12_colab_validation.ipynb`](notebooks/phase12_colab_validation.ipynb)
- **Validation Plan & Tolerances:** [`reports/phase12_colab_validation_plan.md`](reports/phase12_colab_validation_plan.md)
- **Expected Benchmark CSV:** [`experiments/colab/expected_metrics.csv`](experiments/colab/expected_metrics.csv)
- **Machine Specification:** [`experiments/colab/reproducibility_spec.json`](experiments/colab/reproducibility_spec.json)

### Running in Google Colab:
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `notebooks/phase12_colab_validation.ipynb`.
3. Upload `data/features/freight_features_expanded.csv` when prompted in Cell 1.
4. Execute all cells (`Runtime -> Run all`).

---

## 6. Project Directory Structure

```text
ficos/
├── configs/                  # Pipeline, feature, and model configurations
├── data/
│   ├── raw/                  # Immutable raw source files (EIA, World Bank, GPR)
│   ├── processed/            # Canonical 5,145 x 84 master dataset
│   └── features/             # Canonical expanded 3,353 x 299 feature matrix
├── notebooks/                # Exploratory notebooks & Phase 12 Colab validator
│   └── phase12_colab_validation.ipynb
├── src/                      # Core production package
│   ├── data/                 # Ingestion and schema validation
│   ├── features/             # Causal feature engineering pipelines
│   ├── models/               # Persistence, Ridge, XGBoost, PyTorch LSTM
│   ├── decision/             # Vessel feasibility, port constraints, cost & risk models
│   └── evaluation/           # Metric evaluators and walk-forward CV
├── experiments/              # Full experiment artifacts, logs, and figures
│   ├── expanded/             # Phase 11 model audit and decision CSVs
│   └── colab/                # Phase 12 Colab expected benchmarks and specs
├── reports/                  # Detailed engineering and audit reports
├── scripts/                  # Reproducible CLI execution and preparation scripts
└── tests/                    # 134 automated unit, integration, and adversarial tests
```

---

## 7. Getting Started Locally

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/SSOHEB/FICOS-ML.git
cd FICOS-ML

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Running the Test Suite
```bash
pytest -v
```
*Expected: 134 passed, 0 failed (100% green).*

### Generating Decision Recommendations
```bash
python main.py
```

---

## 8. Limitations & Boundaries
- **Route-Level Freight:** FICOS forecasts index-level representative vessel daily hire rates ($/day). Specific fixture pricing will vary by route bunker consumption and canal tolls.
- **Deep Learning Stochasticity:** PyTorch LSTM outputs may vary slightly ($\pm 5\%$) across different GPU CUDA drivers. Ridge and Persistence are mathematically deterministic.
- **Economic Value:** Decision simulated accuracy represents market timing correctness; realized savings depend on specific cargo fixture negotiation.

---

## 9. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
