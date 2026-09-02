# Phase 9 Forecast Uncertainty & Charter Decision Engine Report

## 1. Executive Summary

This report documents the design, mathematical formulation, operational constraints, and historical scenario evaluation of the **Phase 9 Charter Decision-Support Engine** for the **FICOS (Freight Intelligence & Charter Optimization System)** project.

### Core Objective
Transform the empirically selected **Ridge Regression forecasting model ($\alpha=1.0$)** into an explainable, decision-support layer answering the operational question:
> *"I need to move dry-bulk cargo from an origin to an East Coast Indian port. Based on the freight forecast, uncertainty bounds, and operational constraints, should I charter now or wait, which vessel type is suitable, and what risks should I consider?"*

> [!IMPORTANT]
> **PROTOTYPE DECISION-SUPPORT NOTICE:**
> This system is a **prototype decision-support calculation tool**. It is designed to assist human chartering teams with transparent, rule-based scenario intelligence. It does **NOT** automatically execute commercial freight fixtures and does **NOT** claim realized monetary savings for SAIL.

---

## 2. Forecast Uncertainty Quantification Methodology

Ridge regression produces deterministic point estimates $\hat{Y}_{t+h}$. To provide actionable risk bounds without making unfounded Gaussian distribution assumptions, FICOS employs **empirical out-of-sample residual quantile estimation** derived from the Phase 8 walk-forward holdout evaluation.

### Mathematical Formulation
Let $e_t = Y_t - \hat{Y}_t^{\text{Ridge}}$ represent the historical out-of-sample forecasting residual.

From the empirical residual distribution $\mathcal{D}_e$, we compute:
- **$P_{10}$ Error Quantile ($q_{0.10}$)**: 10th percentile empirical residual (downside limit).
- **$P_{50}$ Error Quantile ($q_{0.50}$)**: Median empirical residual ($\approx 0$).
- **$P_{90}$ Error Quantile ($q_{0.90}$)**: 90th percentile empirical residual (upside limit).

For a multi-step forecast horizon step $h \in [1, \dots, H]$ (where $H=7$ trading sessions), the empirical prediction interval expands causally according to diffusion scaling:

$$\text{Lower Bound } P_{10}(h) = \max\left(0, \hat{Y}_{t+h} + q_{0.10} \cdot \sqrt{h}\right)$$

$$\text{Point Forecast } P_{50}(h) = \hat{Y}_{t+h}$$

$$\text{Upper Bound } P_{90}(h) = \hat{Y}_{t+h} + q_{0.90} \cdot \sqrt{h}$$

$$\text{Interval Width}(h) = \text{Upper Bound } P_{90}(h) - \text{Lower Bound } P_{10}(h)$$

### Residual Parameters by Vessel Class:
- **Handysize (`bdi_hsi`)**: $q_{0.10} = -3.20$, $q_{0.90} = +3.35$, $\text{MAE} = 1.97$
- **Supramax (`bdi_si`)**: $q_{0.10} = -6.80$, $q_{0.90} = +7.10$, $\text{MAE} = 4.21$
- **Panamax (`bdi_pi`)**: $q_{0.10} = -16.40$, $q_{0.90} = +17.20$, $\text{MAE} = 10.10$
- **Capesize (`bdi_ci`)**: $q_{0.10} = -98.50$, $q_{0.90} = +105.00$, $\text{MAE} = 63.82$

---

## 3. Forecast Horizon Design

- **Configurable Horizon ($H$)**: **`7` trading sessions** (~1.5 calendar weeks), matching typical spot dry-bulk laycan fixing windows.
- **Trajectory Projection**: Evaluates step-by-step freight momentum, drift rates, and expanding uncertainty cones across days $h \in \{1, 2, 3, 4, 5, 6, 7\}$.

---

## 4. Port Constraints & Operational Assumptions

The system incorporates physical and operational parameters for key East Coast Indian ports.

| Port Name | Max Draft (m) | Max LOA (m) | Max Beam (m) | Max DWT | Allowed Vessels | Discharge Rate (TPD) | Turnaround (Days) | Status |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **Paradip** | 14.5 | 230.0 | 32.5 | 85,000 | Handysize, Supramax, Panamax | 25,000 | 3.5 | *Prototype Assumption* |
| **Vizag** | 16.5 | 280.0 | 45.0 | 150,000 | Handysize, Supramax, Panamax, Capesize | 30,000 | 3.0 | *Prototype Assumption* |
| **Gangavaram** | 19.5 | 330.0 | 55.0 | 200,000 | Handysize, Supramax, Panamax, Capesize | 45,000 | 2.5 | *Prototype Assumption* |
| **Gopalpur** | 12.5 | 200.0 | 30.0 | 55,000 | Handysize, Supramax | 18,000 | 4.0 | *Prototype Assumption* |
| **Dhamra** | 18.0 | 310.0 | 50.0 | 180,000 | Handysize, Supramax, Panamax, Capesize | 40,000 | 2.5 | *Prototype Assumption* |
| **Sagar-Sandheads** | 16.0 | 290.0 | 48.0 | 150,000 | Handysize, Supramax, Panamax, Capesize | 15,000 | 5.0 | *Prototype Assumption* |
| **Haldia** | 8.5 | 190.0 | 28.0 | 35,000 | Handysize *(Shallow Draft Constrained)* | 12,000 | 4.5 | *Prototype Assumption* |

---

## 5. Transparent Cost & Timing Model

For every day $d \in \{0, 1, \dots, H\}$ in the laycan window (where $d=0$ is Charter Now):

$$\text{Total Expected Cost}(d) = \text{Freight Cost}(d) + \text{Port Stay Cost} + \text{Holding Delay Cost}(d)$$

1. **Freight Sea Voyage Cost**:
   $$\text{Freight Cost}(d) = \left(\hat{Y}_{t+d} \times \text{Daily Hire Multiplier}\right) \times \text{Voyage Duration Days}$$
2. **Port Stay & Handling Demurrage**:
   $$\text{Port Stay Cost} = \text{Turnaround Days} \times \text{Daily Demurrage Rate}$$
3. **Holding / Idle Cost**:
   $$\text{Holding Delay Cost}(d) = d \times \text{Daily Cargo Holding Cost ($1,500/day)}$$

### Net Savings & Timing Comparison:
$$\text{Estimated Savings} = \text{Total Cost}(0) - \min_{d \in [1, \dots, H]} \text{Total Cost}(d)$$

---

## 6. Transparent Market-Entry Decision Rules

The decision engine applies explainable, deterministic thresholds:

1. **`CHARTER NOW`**:
   - Triggered when expected freight rates are forecasted to rise by $\ge 1.5\%$ over the laycan horizon, or when market risk is `HIGH` in a non-falling market. Locking in spot rates avoids higher future hire costs.
2. **`WAIT`**:
   - Triggered when expected freight rates ease sufficiently to yield a net cost saving $\ge 2.0\%$ (after accounting for daily cargo holding penalties) and market risk is not `HIGH`.
3. **`FLEXIBLE / MONITOR`**:
   - Triggered when projected freight movements ($< 1.5\%$) remain within standard forecast uncertainty and market noise.

---

## 7. Multi-Factor Risk Model

The risk layer evaluates four contemporaneous historical signals:
1. **Forecast Dispersion**: Uncertainty interval width relative to spot level ($>20\% \to$ High Risk).
2. **Vessel Class Volatility**: Capesize segment structural volatility.
3. **Geopolitical Risk Spike**: GPR ratio relative to 30-day moving average ($>1.30\text{x} \to$ High Risk).
4. **Coastal Weather Alerts**: Extreme precipitation / monsoon disruption ($>25\text{mm} \to$ Weather Alert).

---

## 8. Historical Scenario Replay Evaluation (10 Case Studies)

The decision engine was evaluated on 10 realistic historical shipment inquiries to East Coast Indian ports:

| Scenario ID | Date | Cargo | Quantity | Destination | Vessel | Action | Spot Cost ($) | Optimal Cost ($) | Net Savings ($) | Risk |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SCN_01** | `2016-02-15` | Coking Coal | 75k MT | Paradip | Panamax | **CHARTER NOW** | $127,650 | $127,650 | $0 | HIGH |
| **SCN_02** | `2016-08-10` | Thermal Coal | 55k MT | Vizag | Supramax | **FLEXIBLE** | $179,832 | $179,832 | $0 | HIGH |
| **SCN_03** | `2017-06-20` | Iron Ore | 170k MT | Gangavaram | Capesize | **FLEXIBLE** | $365,800 | $267,318 | $98,482 | HIGH |
| **SCN_04** | `2017-10-16` | Coking Coal | 75k MT | Dhamra | Panamax | **CHARTER NOW** | $397,862 | $397,862 | $0 | MEDIUM |
| **SCN_05** | `2018-04-12` | Limestone | 45k MT | Gopalpur | Supramax | **FLEXIBLE** | $189,960 | $189,960 | $0 | HIGH |
| **SCN_06** | `2018-07-18` | Thermal Coal | 30k MT | Haldia | Handysize | **FLEXIBLE** | $112,320 | $112,320 | $0 | MEDIUM |
| **SCN_07** | `2018-09-25` | Coking Coal | 75k MT | Paradip | Panamax | **CHARTER NOW** | $421,275 | $421,275 | $0 | HIGH |
| **SCN_08** | `2019-02-14` | Iron Ore | 150k MT | Vizag | Panamax | **CHARTER NOW** | $170,062 | $170,062 | $0 | HIGH |
| **SCN_09** | `2019-05-20` | Coking Coal | 70k MT | Dhamra | Panamax | **CHARTER NOW** | $332,600 | $332,600 | $0 | MEDIUM |
| **SCN_10** | `2019-07-10` | Iron Ore | 175k MT | Gangavaram | Capesize | **CHARTER NOW** | $996,080 | $996,080 | $0 | HIGH |

---

## 9. Example Recommendation Output

```json
{
  "charter_action": "CHARTER NOW",
  "recommended_vessel": "Panamax",
  "current_freight_index": 1548.0,
  "optimal_entry_day": 0,
  "expected_cost_now_usd": 397862.50,
  "expected_cost_optimal_usd": 397862.50,
  "estimated_savings_usd": 0.0,
  "estimated_savings_pct": 0.0,
  "risk_level": "MEDIUM",
  "reasons": [
    "Freight rate for Panamax is forecasted to rise by +2.1% over the next 7 trading sessions. Chartering now avoids higher future hire costs.",
    "Recommended vessel 'Panamax' satisfies Dhamra Port navigational constraints (Draft 14.2m <= 18.0m max draft) and provides optimal capacity utilization for 75,000 MT of Coking Coal."
  ],
  "risk_reasons": [
    "Moderate forecast uncertainty (12.4% interval width).",
    "Stable freight conditions with narrow forecast dispersion and baseline risk indicators."
  ],
  "prototype_assumption": true
}
```

---

## 10. Saved Experiment Artifacts

- **Scenario Metadata**: [experiments/phase9/scenarios.csv](file:///c:/Users/soheb/OneDrive/Desktop/ficos/experiments/phase9/scenarios.csv)
- **Recommendations Table**: [experiments/phase9/recommendations.csv](file:///c:/Users/soheb/OneDrive/Desktop/ficos/experiments/phase9/recommendations.csv)
- **Cost Comparisons Table**: [experiments/phase9/cost_comparison.csv](file:///c:/Users/soheb/OneDrive/Desktop/ficos/experiments/phase9/cost_comparison.csv)
- **Visual Diagnostics**: [experiments/phase9/figures/](file:///c:/Users/soheb/OneDrive/Desktop/ficos/experiments/phase9/figures)
  - `01_decision_actions_and_risk.png`
  - `02_charter_cost_comparison.png`
