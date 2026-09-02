# Expanded KOBC Freight Model Retraining Report

## 1. Executive Summary
This experiment retrained four forecasting architectures (**Persistence, Ridge Regression, XGBoost, PyTorch LSTM**) on the genuine 2020+ KOBC freight dataset from `data/features/freight_features_expanded.csv`.

- **Total Usable KOBC Rows:** 1582 (from 2020-02-06 to 2026-08-31)
- **Train Set (70%):** 1107 rows (2020-02-06 to 2024-08-28)
- **Validation Set (15%):** 237 rows (2024-08-29 to 2025-08-28)
- **Final Test Set (15%):** 238 rows (2025-08-29 to 2026-08-31)
- **Engineered Causal Features Used:** 172

---

## 2. Test Set Evaluation Metrics

| Vessel / Target | Model | MAE | RMSE | sMAPE (%) | R² | Directional Accuracy (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Handy** | Persistence | 90.27 | 136.81 | 0.77% | 0.9954 | 0.00% |
| **Handy** | Ridge | 186.83 | 229.13 | 1.57% | 0.9872 | 60.92% |
| **Handy** | XGBoost | 281.81 | 417.35 | 2.07% | 0.9575 | 54.20% |
| **Handy** | LSTM | 2024.09 | 2665.43 | 13.80% | -0.6785 | 50.66% |
| **Supramax** | Persistence | 166.56 | 242.57 | 0.99% | 0.9908 | 0.00% |
| **Supramax** | Ridge | 178.88 | 233.08 | 1.02% | 0.9915 | 62.18% |
| **Supramax** | XGBoost | 273.97 | 341.34 | 1.52% | 0.9818 | 62.61% |
| **Supramax** | LSTM | 2111.72 | 2804.50 | 10.41% | -0.1799 | 57.21% |
| **Panamax** | Persistence | 256.78 | 346.25 | 1.46% | 0.9831 | 0.00% |
| **Panamax** | Ridge | 238.48 | 321.35 | 1.34% | 0.9854 | 63.87% |
| **Panamax** | XGBoost | 254.28 | 334.10 | 1.45% | 0.9842 | 63.45% |
| **Panamax** | LSTM | 2008.20 | 2512.97 | 10.38% | 0.1288 | 55.02% |
| **Cape** | Persistence | 1131.71 | 1447.10 | 3.19% | 0.9682 | 0.00% |
| **Cape** | Ridge | 1098.31 | 1433.56 | 3.09% | 0.9688 | 62.61% |
| **Cape** | XGBoost | 1327.87 | 1832.60 | 3.57% | 0.9490 | 59.24% |
| **Cape** | LSTM | 3352.38 | 4057.19 | 9.44% | 0.7494 | 55.02% |

---

## 3. Walk-Forward Cross-Validation (5 Folds Aggregate)

| Vessel | Model | Mean MAE ± Std | Mean RMSE ± Std | Mean sMAPE ± Std (%) | Mean DA ± Std (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Cape** | LSTM | 2699.48 ± 1412.18 | 3223.38 ± 1526.30 | 9.67% ± 3.15% | 49.32% ± 2.80% |
| **Cape** | Persistence | 1006.02 ± 219.00 | 1305.33 ± 243.13 | 3.73% ± 0.61% | 0.16% ± 0.35% |
| **Cape** | Ridge | 975.75 ± 179.69 | 1250.91 ± 236.38 | 3.78% ± 0.82% | 61.95% ± 10.97% |
| **Cape** | XGBoost | 1028.86 ± 379.13 | 1336.70 ± 499.81 | 3.72% ± 0.65% | 63.36% ± 8.23% |
| **Handy** | LSTM | 1249.82 ± 480.89 | 1469.74 ± 542.43 | 10.47% ± 2.74% | 54.90% ± 13.50% |
| **Handy** | Persistence | 94.49 ± 14.54 | 134.96 ± 15.49 | 0.92% ± 0.14% | 0.16% ± 0.35% |
| **Handy** | Ridge | 115.47 ± 22.64 | 148.89 ± 24.89 | 1.12% ± 0.26% | 63.97% ± 7.08% |
| **Handy** | XGBoost | 182.62 ± 116.99 | 242.93 ± 139.30 | 1.60% ± 0.66% | 55.14% ± 4.49% |
| **Panamax** | LSTM | 1546.19 ± 525.34 | 1833.85 ± 547.52 | 10.03% ± 4.33% | 45.08% ± 12.25% |
| **Panamax** | Persistence | 236.06 ± 52.86 | 318.23 ± 69.35 | 1.58% ± 0.29% | 0.47% ± 0.70% |
| **Panamax** | Ridge | 229.94 ± 31.62 | 303.53 ± 33.78 | 1.58% ± 0.26% | 64.61% ± 4.76% |
| **Panamax** | XGBoost | 249.62 ± 38.28 | 330.11 ± 46.80 | 1.77% ± 0.55% | 62.42% ± 9.77% |
| **Supramax** | LSTM | 894.26 ± 521.51 | 1177.50 ± 734.21 | 5.63% ± 2.66% | 54.42% ± 9.79% |
| **Supramax** | Persistence | 152.71 ± 24.86 | 206.71 ± 38.83 | 1.04% ± 0.11% | 0.16% ± 0.35% |
| **Supramax** | Ridge | 170.29 ± 58.07 | 210.98 ± 64.67 | 1.13% ± 0.35% | 66.19% ± 5.24% |
| **Supramax** | XGBoost | 191.14 ± 83.07 | 247.37 ± 100.78 | 1.20% ± 0.33% | 66.99% ± 8.80% |

---

## 4. Model Selection & Champion Determination
Champion models per target are selected based on lowest MAE/sMAPE and superior directional accuracy across both holdout test and expanding walk-forward folds.
