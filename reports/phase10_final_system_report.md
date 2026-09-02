# FICOS Phase 10 Final System Integration & Production-Ready MVP Report

## 1. Problem Statement

Steel Authority of India Limited (SAIL) and major domestic industrial enterprises import tens of millions of metric tons of metallurgical coking coal, thermal coal, limestone, and iron ore from overseas origins (Australia, South Africa, Indonesia, Brazil, UAE) to East Coast Indian ports (**Paradip**, **Visakhapatnam**, **Gangavaram**, **Gopalpur**, **Dhamra**, **Sagar-Sandheads**, **Haldia**).

International dry-bulk freight rates fluctuate volatilely driven by macroeconomic shocks, commodity demand cycles, bunker fuel costs, and geopolitical events. Furthermore, physical port constraints (shallow draft, LOA limits, berth congestion, handling rates) dictate which vessel types (**Handysize**, **Supramax**, **Panamax**, **Capesize**) can berth.

Chartering teams face a critical operational question:
> *"I have a cargo parcel of $N$ metric tons to transport to an East Coast Indian port. Based on current market levels, freight forecasts, uncertainty bounds, and operational constraints, should I charter a vessel now, wait for a lower rate, or monitor the market, which vessel size is optimal, and what risks must be managed?"*

**FICOS (Freight Intelligence & Charter Optimization System)** solves this challenge by providing an end-to-end, explainable, decision-support platform bridging:
$$\text{DATA} \to \text{FEATURES} \to \text{FORECAST} \to \text{UNCERTAINTY} \to \text{VESSEL/PORT} \to \text{COST} \to \text{RISK} \to \text{DECISION} \to \text{AUDIT}$$

---

## 2. Final Architecture Overview

The system is structured into a modular, testable pipeline:
1. **Data Ingestion & Cleaning**: Ingests multi-source market, macro, and meteorological time-series into a consolidated, clean master dataset.
2. **Causal Feature Engineering**: Constructs 135 causal variables with strictly causal lag alignment ($t \to t+1$).
3. **Forecasting Engine**: Production Ridge regression forecaster ($\alpha=1.0$) generating 1-step and 7-day multi-step index projections.
4. **Empirical Uncertainty Engine**: Quantifies non-parametric $P_{10}, P_{50}, P_{90}$ prediction intervals from out-of-sample residuals.
5. **Operational Suitability Engine**: Evaluates port navigational limits and selects optimal vessel classes.
6. **Transparent Cost Model**: Computes freight sea hire, port stay demurrage, and cargo holding delay costs.
7. **Multi-Factor Risk Engine**: Evaluates volatility, forecast dispersion, GPR spikes, and coastal weather alerts.
8. **Charter Decision Engine**: Applies transparent rule-based logic to recommend `CHARTER NOW`, `WAIT`, or `FLEXIBLE / MONITOR`.
9. **Application Service & CLI/API Layer**: Unified `FICOSService` with command-line interface, REST endpoints, and reproducible audit logs.

---

## 3. Data Pipeline & Quality Assurance (Phase 2)

- **Master Dataset Path**: `data/processed/master_dataset.csv`
- **Temporal Span**: `2012-08-01` to `2019-07-31` (2,556 continuous calendar days, 1,749 trading sessions).
- **Quality Verification**:
  - Exactly zero duplicate timestamps.
  - Strictly chronological date ordering.
  - Zero missing values on Baltic Exchange trading sessions across all 4 target sub-indices (`bdi_hsi`, `bdi_si`, `bdi_pi`, `bdi_ci`).
  - No synthetic data fabrication.

---

## 4. Causal Feature Engineering (Phase 4)

- **Feature Matrix Path**: `data/features/freight_features.csv` (1,749 trading days $\times$ 141 columns).
- **135 Causal Features across 7 Families**:
  1. *Autoregressive Lags (28)*: Lags $t-1, t-2, t-3, t-5, t-10, t-15, t-21$.
  2. *Cross-Vessel Lags (28)*: Lags $t-1, t-5, t-10, t-21$ across cross-vessel series.
  3. *Short-Term Returns & Differences (24)*: 1-day, 5-day, 10-day differences and percent returns.
  4. *Rolling Statistics & Volatility (28)*: 7-day, 14-day, 21-day rolling means, standard deviations, and return volatilities.
  5. *Macroeconomic Exogenous Variables (12)*: WTI crude, Brent crude, USD/INR exchange rate, Baltic Clean Tanker Index, GSCPI supply chain pressure.
  6. *Geopolitical Risk (GPR) Metrics (9)*: Daily GPR levels, moving averages, and spike ratios.
  7. *Calendar & Meteorological Variables (6)*: Day of week, month, quarter, precipitation, temperature, wind speed.
- **Zero Future Leakage**: Mathematically verified that feature vectors at time $t$ predict target values at $t+1$ with zero future overlap.

---

## 5. Forecasting Models Evaluated (Phases 5–7)

1. **Naive Persistence**: $\hat{Y}_{t+1} = Y_t$
2. **Moving Averages**: Windows $W \in \{3, 5, 10, 21\}$
3. **Ridge Regularized Linear Regression**: Regularization $\alpha \in \{0.1, 1.0, 10.0, 100.0\}$
4. **XGBoost (Gradient Boosted Trees)**: Nonlinear tree ensemble with shrinkage and feature subsampling.
5. **PyTorch LSTM**: Recurrent deep learning model with 21-day sliding lookback sequences.
6. **Equal-Weighted Ensemble**: $0.5 \times \text{Ridge} + 0.5 \times \text{XGBoost}$.

---

## 6. Why Ridge Regression Was Selected as Champion

In 5-fold expanding-window walk-forward validation across 1,000 out-of-sample trading days:
- **Lowest Out-of-Sample MAE**:
  - Handysize: **1.97** (vs Persistence 3.45, XGBoost 6.85, LSTM 22.34)
  - Supramax: **4.21** (vs Persistence 7.11, XGBoost 17.96, LSTM 29.81)
  - Panamax: **10.10** (vs Persistence 18.65, XGBoost 23.62, LSTM 52.76)
  - Capesize: **63.82** (vs Persistence 67.95, XGBoost 79.75, LSTM 162.17)
- **Highest Directional Accuracy**: ~**79% to 80%** across all vessel segments.
- **Lowest Error Variance Across Regimes**: $\sigma_{\text{MAE}} = 0.34$ on HSI, $0.60$ on SI, $1.42$ on PI, $8.61$ on CI.

---

## 7. What Happened with XGBoost

- **Mechanism**: Decision trees perform axis-aligned orthogonal feature splits, producing piecewise-constant step predictions.
- **Outcome**: On smooth, continuous unit-root time series ($ACF_1 > 0.97$), decision trees suffer from quantization error and cannot extrapolate outside historical training ranges during sudden bull expansions or supply squeezes.
- **Status**: Retained as a baseline research candidate.

---

## 8. What Happened with LSTM

- **Mechanism**: Recurrent neural network with memory cells and hidden states.
- **Outcome**: With ~1,350 daily training sequences, recurrent memory gating acted as an adaptive smoothing filter that lagged behind rapid freight turning points. It suffered from parameter overcapacity and high cross-fold variance ($\sigma_{\text{MAE}} = 70.62$ on Capesize).
- **Status**: Retained as a research candidate.

---

## 9. What Happened with the Ensemble

- **Mechanism**: Evaluated an equal-weighted ensemble of Ridge + XGBoost ($\hat{Y}_{\text{ens}} = 0.5 \hat{Y}_{\text{Ridge}} + 0.5 \hat{Y}_{\text{XGBoost}}$) walk-forward.
- **Outcome**: The ensemble performed 2nd best on Capesize (MAE 65.63) and Panamax (MAE 15.23), beating standalone XGBoost, but **did not beat standalone Ridge** (63.82 and 10.10). Blending the noisier tree predictions degraded Ridge's precision.
- **Decision**: **Ensemble was rejected**.

---

## 10. Walk-Forward Validation Methodology (Phase 8)

- **Expanding Window**: 5 chronological folds evaluated across 4 years (2015 to 2019).
- **Evaluation Size**: 200 trading sessions (~9.5 months) per fold.
- **Tested Regimes**: 2015–2016 market crash, 2016–2017 recovery, 2017 bull expansion, 2018 tariff volatility, and 2019 Brumadinho supply shock.

---

## 11. Uncertainty Quantification Methodology (Phase 9)

- Non-parametric empirical prediction intervals derived from walk-forward residual error distributions:
  $$\text{Lower Bound } P_{10}(h) = \max\left(0, \hat{Y}_{t+h} + q_{0.10} \cdot \sqrt{h}\right)$$
  $$\text{Upper Bound } P_{90}(h) = \hat{Y}_{t+h} + q_{0.90} \cdot \sqrt{h}$$
- Avoids fragile Gaussian assumptions and provides transparent risk bands.

---

## 12. Vessel & Port Navigational Feasibility

- Ports Configured: **Paradip**, **Visakhapatnam**, **Gangavaram**, **Gopalpur**, **Dhamra**, **Sagar-Sandheads**, **Haldia**.
- Restricts vessels based on physical draft, LOA, beam, and DWT (e.g. Haldia restricted to Handysize due to 8.5m max draft; deepwater Gangavaram accommodates Capesize up to 19.5m draft).

---

## 13. Transparent Voyage Cost Model

$$\text{Total Expected Cost}(d) = \text{Freight Hire Cost}(d) + \text{Port Stay Demurrage} + \text{Holding Delay Cost}(d)$$
- Daily Hire Rate = Index Level $\times$ Vessel Multiplier ($10.0$ to $15.0$/day).
- Port Demurrage = Turnaround Days $\times$ Daily Demurrage Rate ($10k to $22k/day).
- Cargo Holding Cost = Days Waited $\times$ $1,500/day.

---

## 14. Multi-Factor Risk Model

Evaluates contemporaneous signals into `LOW`, `MEDIUM`, or `HIGH` risk:
1. Forecast uncertainty ratio ($>20\%$).
2. Structural segment volatility (Capesize).
3. Geopolitical risk spike ratio ($>1.30\text{x}$).
4. Destination port approach extreme precipitation ($>25\text{mm}$).

---

## 15. Charter Decision Logic

- **`CHARTER NOW`**: Projected freight rise $\ge +1.5\%$ or `HIGH` risk in non-falling market.
- **`WAIT`**: Projected decline yields net cost savings $\ge 2.0\%$ net of daily holding penalties.
- **`FLEXIBLE / MONITOR`**: Projected movement within uncertainty noise band ($< 1.5\%$).

---

## 16. Historical Scenario Replay Methodology

- Evaluated 10 historical shipment inquiries spanning 2016 through 2019.
- Uses strictly contemporaneous information available on or before decision date $T$ ($t \le T$).

---

## 17. Hindsight Benchmark vs Decision-Time Knowledge

- **Decision-Time Achievable Result**: What the system recommended at decision date $T$ using solely prior information.
- **Hindsight Benchmark**: Retrospective oracle computed for post-hoc regret analysis, clearly labeled as retrospective hindsight and never accessible to the live decision engine.

---

## 18. End-to-End Execution Demonstration

Running:
```bash
python -m src.application --quantity 75000 --cargo "Coking Coal" --destination dhamra --date 2018-09-25
```
Yields:
```text
============================================================
           FICOS CHARTER DECISION RECOMMENDATION
============================================================
Decision Date:       2018-09-25
Destination Port:    DHAMRA
Cargo:               75,000 MT of Coking Coal
Recommended Vessel:  Panamax
Feasible Vessels:    Handysize, Supramax, Panamax
------------------------------------------------------------
7-DAY FREIGHT RATE FORECAST & UNCERTAINTY (Points):
  Day 1: Point= 1661.8 | 80% CI=[ 1647.0 to  1678.9]
  Day 2: Point= 1684.9 | 80% CI=[ 1663.9 to  1709.2]
  Day 3: Point= 1708.4 | 80% CI=[ 1682.6 to  1738.0]
  Day 4: Point= 1732.1 | 80% CI=[ 1702.4 to  1766.4]
  Day 5: Point= 1756.2 | 80% CI=[ 1723.0 to  1794.5]
  Day 6: Point= 1780.7 | 80% CI=[ 1744.3 to  1822.6]
  Day 7: Point= 1805.4 | 80% CI=[ 1766.1 to  1850.8]
------------------------------------------------------------
DECISION:            >>> CHARTER NOW <<<
Expected Cost (Now): $418,775.00
Optimal Cost:        $418,775.00
------------------------------------------------------------
MARKET RISK LEVEL:   HIGH
  * Elevated historical freight return volatility.
  * Elevated Geopolitical Risk ratio (1.19x).
------------------------------------------------------------
DECISION RATIONALE:
  * CHARTER NOW because Panamax freight rate is forecast to rise by +10.2% over the 7-day horizon. Fixing now avoids higher future hire rates.
  * Recommended 'Panamax' as optimal capacity utilization for 75,000 MT satisfying Dhamra Port max draft limit of 18.0m.
============================================================
```

---

## 19. Key Limitations & Assumptions

1. **Historical Data Scope**: System is trained and validated on historical data (2012 to 2019); it does not ingest live real-time feeds in this offline MVP.
2. **Prototype Port Assumptions**: Port operational limits and demurrage values in `configs/ports.yaml` are calibrated prototype assumptions for decision modeling, not official hydrographic or terminal regulations.
3. **Decision-Support Boundary**: The engine is designed for decision support; it does not execute financial transactions or autonomous vessel bookings.

---

## 20. Future Extensions

1. Integration with real-time AIS vessel position tracking and live Baltic Exchange fixtures.
2. Direct API connectors to port authority vessel queue management systems.
3. Dynamic bunker fuel price hedging integration.

---

## 21. SIH Demonstration Workflow

1. **CLI Demo**: Run `python -m src.application --quantity 75000 --destination dhamra --date 2018-09-25` to show live explainable charter recommendations.
2. **REST API Demo**: Run `src/application/api.py` to demonstrate `GET /health` and `POST /recommend` integration for web dashboards.
3. **Audit Verification**: Inspect `experiments/phase10/audit_examples.json` to prove zero future leakage and full reproducibility.
