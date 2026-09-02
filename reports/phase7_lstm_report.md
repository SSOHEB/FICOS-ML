# Phase 7 LSTM Forecasting & Multi-Model Comparative Evaluation Report

## 1. Executive Summary

This report documents the implementation, chronological evaluation, and empirical comparison of a **Long Short-Term Memory (LSTM) Deep Learning Forecaster** for the 4 Baltic dry-bulk freight sub-indices (`HSI`, `SI`, `PI`, `CI`) against the benchmarks established in Phase 5 and Phase 6 (Naive Persistence, Moving Average, Ridge Regression, and XGBoost).

### Core Research Question:
> **"Can a recurrent deep learning architecture (LSTM) using a 21-day sliding historical lookback improve upon the regularized linear (Ridge) and tree-based (XGBoost) models in 1-step-ahead freight rate forecasting?"**

---

## 2. Sequence Construction & Input Design

### 1. Sliding Lookback Windows
- **Lookback Window ($W$)**: **`21` consecutive trading sessions** ($t - 20, \dots, t$).
- **Input Tensor Shape**: $(N, 21, 61)$ where $N$ is the number of valid sequences.
- **Target**: Next observed Baltic Dry Index trading session value ($Y_{t+1}$).
- **Sequence Causality**: Sequence ending at index $t$ uses only information $\le t$ to forecast $t+1$. Continuous sequence generation across the train/test boundary prevents boundary distortion.

### 2. Controlled Feature Subset (61 Selected Variables)
Rather than passing all 135 features into recurrent cells, a parsimonious subset was curated:
1. **Base Freight Levels (4)**: `bdi_hsi_level`, `bdi_si_level`, `bdi_pi_level`, `bdi_ci_level`
2. **Autoregressive Lags (16)**: Lags $t-1, t-2, t-3, t-5$ across all 4 indices
3. **Cross-Vessel Lags (8)**: `cross_bdi_*_lag_1`, `cross_bdi_*_lag_5`
4. **Short Momentum & Differences (8)**: `bdi_*_diff_1`, `bdi_*_pct_change_1`
5. **Rolling Statistics & Volatility (20)**: `bdi_*_roll_mean_7`, `bdi_*_roll_std_7`, `bdi_*_return_vol_7`
6. **Exogenous Macro & Shocks (5)**: `wti_usd_bbl_lag_1`, `brent_usd_bbl_lag_1`, `usd_inr_lag_1`, `gpr_lag_1`, `gpr_spike_ratio_ma30`

---

## 3. Training Strategy & Architecture

```yaml
lstm:
  lookback: 21
  hidden_size: 64
  dense_units: 32
  num_layers: 1
  dropout: 0.15
  learning_rate: 0.001
  weight_decay: 0.0001
  batch_size: 32
  max_epochs: 60
  early_stopping_patience: 10
  val_ratio: 0.15 (chronological out-of-time split from training partition)
  random_seed: 42
```

### Strict Train-Only Preprocessing
- **Feature Scaler**: `StandardScaler` fitted **exclusively on the 1,381 training sessions**.
- **Target Scaler**: `StandardScaler` fitted **exclusively on the training target** and inverted at inference time ($\hat{y} = \text{Scaler}_y^{-1}(\hat{z})$).
- **Early Stopping**: Monitored on the final 15% out-of-time slice of the training partition to prevent overfitting.

---

## 4. Comprehensive Multi-Model Benchmark (Test Period: 346 Trading Days)

### Empirical Results Table

| Target Segment | Model Family | MAE | RMSE | sMAPE (%) | $R^2$ | Directional Accuracy (%) | Overall Ranking |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`bdi_hsi`** (Handysize) | **Ridge ($\alpha=1.0$)** | **2.12** | **2.69** | **0.45%** | **0.9993** | **77.46%** | **1st (Best)** |
| | Naive Persistence | 3.30 | 4.71 | 0.70% | 0.9979 | 10.98% | 2nd |
| | XGBoost | 4.25 | 5.60 | 0.83% | 0.9971 | 63.87% | 3rd |
| | LSTM (Lookback=21) | 22.99 | 27.25 | 4.40% | 0.9304 | 48.55% | 4th |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`bdi_si`** (Supramax) | **Ridge ($\alpha=1.0$)** | **4.15** | **5.52** | **0.50%** | **0.9991** | **77.46%** | **1st (Best)** |
| | Naive Persistence | 7.60 | 10.82 | 0.95% | 0.9964 | 5.20% | 2nd |
| | XGBoost | 10.00 | 14.54 | 1.19% | 0.9935 | 69.08% | 3rd |
| | LSTM (Lookback=21) | 20.56 | 24.79 | 2.39% | 0.9812 | 44.22% | 4th |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`bdi_pi`** (Panamax) | **Ridge ($\alpha=1.0$)** | **10.62** | **14.14** | **0.85%** | **0.9980** | **78.61%** | **1st (Best)** |
| | XGBoost | 16.24 | 32.21 | 1.13% | 0.9899 | 76.01% | 2nd |
| | Naive Persistence | 19.97 | 27.47 | 1.58% | 0.9926 | 1.45% | 3rd |
| | LSTM (Lookback=21) | 89.34 | 114.64 | 6.70% | 0.8715 | 60.40% | 4th |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`bdi_ci`** (Capesize) | **Ridge ($\alpha=1.0$)** | **73.34** | **96.05** | **5.44%** | **0.9902** | **71.10%** | **1st (Best)** |
| | XGBoost | 77.43 | 102.69 | 6.76% | 0.9888 | 69.08% | 2nd |
| | Naive Persistence | 80.57 | 110.77 | 5.59% | 0.9870 | 0.00% | 3rd |
| | LSTM (Lookback=21) | 171.54 | 212.11 | 15.92% | 0.9524 | 54.05% | 4th |

---

## 5. Technical Diagnostic: Why LSTM Underperformed on Raw Freight Levels

1. **Sample Size vs. Parameter Capacity**:
   - The training set contains 1,381 daily sessions (~1,350 lookback sequences).
   - Recurrent architectures have hundreds of gating parameters. With limited cyclical repetitions, recurrent hidden states overfit to the training regime and experience drift when market regimes shift out-of-sample.
2. **Phase Lag on Unit-Root Persistence Series**:
   - Dry-bulk freight rates follow near random-walk persistence ($ACF_1 > 0.97$).
   - The recurrent memory gates act analogously to an adaptive smoothing filter, lagging behind rapid market turns and accumulating phase lag errors.
3. **Linear Efficiency**:
   - Ridge regression directly maps contemporaneous levels and lag differences into an explicit continuous 1-step forecast ($\hat{Y}_{t+1} = \beta_0 + 0.98 Y_t + \dots$) with minimal estimation variance.

---

## 6. Saved Model Checkpoints & Artifacts

- **Model Checkpoints**: [experiments/phase7/models/](file:///c:/Users/soheb/OneDrive/Desktop/ficos/experiments/phase7/models)
  - `lstm_bdi_hsi.pt`
  - `lstm_bdi_si.pt`
  - `lstm_bdi_pi.pt`
  - `lstm_bdi_ci.pt`
- **Metrics Table**: [experiments/phase7/metrics.csv](file:///c:/Users/soheb/OneDrive/Desktop/ficos/experiments/phase7/metrics.csv)
- **Predictions Table**: [experiments/phase7/predictions.csv](file:///c:/Users/soheb/OneDrive/Desktop/ficos/experiments/phase7/predictions.csv)
- **Visual Diagnostics**: [experiments/phase7/figures/](file:///c:/Users/soheb/OneDrive/Desktop/ficos/experiments/phase7/figures)
  - `01_actual_vs_all_models_test.png`
  - `02_lstm_residuals_diagnostics.png`
  - `03_lstm_training_history.png`

---

## 7. Key Findings & Next Steps

### Empirical Conclusion:
1. **Ridge Regression remains the best overall forecasting model** across all 4 vessel sub-indices, achieving both the lowest MAE/RMSE and the highest Directional Accuracy (~77-79%).
2. **XGBoost performs second-best**, beating Persistence on Panamax and Capesize but losing to Ridge due to piecewise constant step quantization.
3. **LSTM on raw price levels is not competitive** with linear regularization on daily frequency data.

### Strategic Roadmap for Walk-Forward Model Selection:
- Future modeling should focus on **Ridge regression**, **stationary target differencing ($\Delta Y_{t+1}$)**, **hybrid linear + tree residual boosting**, and **expanding-window walk-forward validation** across multiple historical market cycles.
