import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.schemas import DATE_COLUMN
from src.models.baselines import PersistenceForecaster
from src.models.ridge import RidgeForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.lstm_model import LSTMForecaster
from src.models.evaluation import compute_regression_metrics


KOBC_TARGETS = {
    "target_kobc_handy_next": "kobc_handy_level",
    "target_kobc_supramax_next": "kobc_supramax_level",
    "target_kobc_panamax_next": "kobc_panamax_level",
    "target_kobc_cape_next": "kobc_cape_level",
}


def get_valid_kobc_features(df_kobc: pd.DataFrame) -> List[str]:
    """Select legitimate causal features for the KOBC regime."""
    target_cols = [c for c in df_kobc.columns if c.startswith("target_")]
    meta_cols = [
        "date", "is_baltic_regime", "is_kobc_regime",
        "is_bdi_trading_day", "is_kobc_trading_day", "freight_source"
    ]
    baltic_cols = [c for c in df_kobc.columns if c.startswith("bdi_") or "bdi" in c]
    exclude = set(target_cols + meta_cols + baltic_cols)

    candidate_features = [c for c in df_kobc.columns if c not in exclude]
    # Filter features that have excessive NaNs in the KOBC era (> 20%)
    valid_features = [
        c for c in candidate_features
        if df_kobc[c].isnull().sum() / len(df_kobc) < 0.20
    ]
    return sorted(valid_features)


def run_expanded_experiment():
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "experiments" / "expanded"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    input_path = project_root / "data" / "features" / "freight_features_expanded.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    print(f"Loading expanded features from: {input_path}")
    df = pd.read_csv(input_path)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    # Filter to genuine KOBC period
    df_kobc = df[df[DATE_COLUMN] >= pd.Timestamp("2020-01-01")].sort_values(by=DATE_COLUMN).reset_index(drop=True)
    print(f"Total KOBC rows: {len(df_kobc)} ({df_kobc[DATE_COLUMN].min().strftime('%Y-%m-%d')} to {df_kobc[DATE_COLUMN].max().strftime('%Y-%m-%d')})")

    features = get_valid_kobc_features(df_kobc)
    print(f"Selected {len(features)} valid causal KOBC features.")

    # Drop cold start (first 21 rows for lag-21) and drop terminal row (target is unobserved for final day)
    df_clean = df_kobc.iloc[21:-1].reset_index(drop=True)
    n = len(df_clean)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val

    train_df = df_clean.iloc[:n_train].reset_index(drop=True)
    val_df = df_clean.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df_clean.iloc[n_train + n_val:].reset_index(drop=True)

    print(f"Train: {len(train_df)} rows ({train_df[DATE_COLUMN].min().strftime('%Y-%m-%d')} to {train_df[DATE_COLUMN].max().strftime('%Y-%m-%d')})")
    print(f"Val:   {len(val_df)} rows ({val_df[DATE_COLUMN].min().strftime('%Y-%m-%d')} to {val_df[DATE_COLUMN].max().strftime('%Y-%m-%d')})")
    print(f"Test:  {len(test_df)} rows ({test_df[DATE_COLUMN].min().strftime('%Y-%m-%d')} to {test_df[DATE_COLUMN].max().strftime('%Y-%m-%d')})")

    # Metrics container
    all_metrics = []
    predictions_dict = {
        DATE_COLUMN: test_df[DATE_COLUMN].dt.strftime("%Y-%m-%d").values
    }

    trained_models = {}

    for target_col, current_col in KOBC_TARGETS.items():
        vessel_name = target_col.replace("target_kobc_", "").replace("_next", "").capitalize()
        print(f"\n==================== Training & Evaluating: {vessel_name} ({target_col}) ====================")

        X_train, y_train = train_df[features], train_df[target_col]
        X_val, y_val = val_df[features], val_df[target_col]
        X_test, y_test = test_df[features], test_df[target_col]

        predictions_dict[f"actual_{target_col}"] = y_test.values

        # 1. Persistence Baseline
        p_model = PersistenceForecaster()
        pred_p = p_model.predict(test_df[current_col])
        m_p = compute_regression_metrics(y_test, pred_p, test_df[current_col])
        m_p.update({"model": "Persistence", "target": target_col, "vessel": vessel_name, "split": "test"})
        all_metrics.append(m_p)
        predictions_dict[f"pred_persistence_{target_col}"] = pred_p
        print(f"  Persistence: MAE={m_p['mae']}, RMSE={m_p['rmse']}, sMAPE={m_p['smape']}%, DA={m_p['da_pct']}%")

        # 2. Ridge Regression
        ridge = RidgeForecaster(alpha=1.0, scale_features=True)
        ridge.fit(X_train, y_train)
        pred_r = ridge.predict(X_test)
        m_r = compute_regression_metrics(y_test, pred_r, test_df[current_col])
        m_r.update({"model": "Ridge", "target": target_col, "vessel": vessel_name, "split": "test"})
        all_metrics.append(m_r)
        predictions_dict[f"pred_ridge_{target_col}"] = pred_r
        trained_models[f"ridge_{target_col}"] = ridge
        print(f"  Ridge:       MAE={m_r['mae']}, RMSE={m_r['rmse']}, sMAPE={m_r['smape']}%, DA={m_r['da_pct']}%")

        # 3. XGBoost
        xgb = XGBoostForecaster(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred_x = xgb.predict(X_test)
        m_x = compute_regression_metrics(y_test, pred_x, test_df[current_col])
        m_x.update({"model": "XGBoost", "target": target_col, "vessel": vessel_name, "split": "test"})
        all_metrics.append(m_x)
        predictions_dict[f"pred_xgboost_{target_col}"] = pred_x
        trained_models[f"xgboost_{target_col}"] = xgb
        print(f"  XGBoost:     MAE={m_x['mae']}, RMSE={m_x['rmse']}, sMAPE={m_x['smape']}%, DA={m_x['da_pct']}%")

        # 4. PyTorch LSTM
        lookback = 10
        lstm = LSTMForecaster(
            lookback=lookback, hidden_size=32, dense_units=16,
            learning_rate=0.001, max_epochs=30, random_seed=42
        )
        lstm.fit(X_train, y_train, verbose=False)
        pred_l_raw = lstm.predict(X_test)
        
        # Align LSTM predictions with test_df (initial lookback-1 rows in test)
        # Pad initial lookback-1 elements with persistence to maintain exact length
        pad_count = lookback - 1
        initial_pad = test_df[current_col].iloc[:pad_count].values
        pred_l = np.concatenate([initial_pad, pred_l_raw])
        
        m_l = compute_regression_metrics(y_test.iloc[pad_count:], pred_l_raw, test_df[current_col].iloc[pad_count:])
        m_l.update({"model": "LSTM", "target": target_col, "vessel": vessel_name, "split": "test"})
        all_metrics.append(m_l)
        predictions_dict[f"pred_lstm_{target_col}"] = pred_l
        trained_models[f"lstm_{target_col}"] = lstm
        print(f"  LSTM:        MAE={m_l['mae']}, RMSE={m_l['rmse']}, sMAPE={m_l['smape']}%, DA={m_l['da_pct']}%")

    # Save metrics and predictions
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to: {metrics_path}")

    predictions_df = pd.DataFrame(predictions_dict)
    predictions_path = output_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to: {predictions_path}")

    # ==================== WALK-FORWARD CROSS-VALIDATION ====================
    print("\n==================== Running 5-Fold Expanding Walk-Forward CV ====================")
    fold_metrics = []
    n_total = len(df_clean)
    train_start_ratio = 0.60
    step_ratio = (1.0 - train_start_ratio) / 5  # 0.08 per fold

    for fold_idx in range(5):
        train_end_idx = int(n_total * (train_start_ratio + fold_idx * step_ratio))
        test_end_idx = int(n_total * (train_start_ratio + (fold_idx + 1) * step_ratio))
        if fold_idx == 4:
            test_end_idx = n_total

        fold_train = df_clean.iloc[:train_end_idx].reset_index(drop=True)
        fold_test = df_clean.iloc[train_end_idx:test_end_idx].reset_index(drop=True)

        for target_col, current_col in KOBC_TARGETS.items():
            vessel_name = target_col.replace("target_kobc_", "").replace("_next", "").capitalize()
            X_tr, y_tr = fold_train[features], fold_train[target_col]
            X_te, y_te = fold_test[features], fold_test[target_col]

            # Persistence
            pred_p = PersistenceForecaster().predict(fold_test[current_col])
            mp = compute_regression_metrics(y_te, pred_p, fold_test[current_col])
            mp.update({"fold": fold_idx + 1, "model": "Persistence", "target": target_col, "vessel": vessel_name, "train_size": len(fold_train), "test_size": len(fold_test)})
            fold_metrics.append(mp)

            # Ridge
            r_fold = RidgeForecaster(alpha=1.0).fit(X_tr, y_tr)
            pred_r = r_fold.predict(X_te)
            mr = compute_regression_metrics(y_te, pred_r, fold_test[current_col])
            mr.update({"fold": fold_idx + 1, "model": "Ridge", "target": target_col, "vessel": vessel_name, "train_size": len(fold_train), "test_size": len(fold_test)})
            fold_metrics.append(mr)

            # XGBoost
            x_fold = XGBoostForecaster(n_estimators=100, max_depth=3, learning_rate=0.05).fit(X_tr, y_tr)
            pred_x = x_fold.predict(X_te)
            mx = compute_regression_metrics(y_te, pred_x, fold_test[current_col])
            mx.update({"fold": fold_idx + 1, "model": "XGBoost", "target": target_col, "vessel": vessel_name, "train_size": len(fold_train), "test_size": len(fold_test)})
            fold_metrics.append(mx)

            # LSTM
            l_fold = LSTMForecaster(lookback=10, hidden_size=32, dense_units=16, max_epochs=20).fit(X_tr, y_tr)
            pred_l = l_fold.predict(X_te)
            ml = compute_regression_metrics(y_te.iloc[9:], pred_l, fold_test[current_col].iloc[9:])
            ml.update({"fold": fold_idx + 1, "model": "LSTM", "target": target_col, "vessel": vessel_name, "train_size": len(fold_train), "test_size": len(fold_test)})
            fold_metrics.append(ml)

        print(f"Completed Fold {fold_idx + 1}/5 (Train: {len(fold_train)}, Test: {len(fold_test)})")

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_path = output_dir / "fold_metrics.csv"
    fold_metrics_df.to_csv(fold_path, index=False)
    print(f"Saved fold metrics to: {fold_path}")

    # ==================== FIGURES GENERATION ====================
    print("\nGenerating figures...")
    # 1. Prediction trajectories for each target
    for target_col in KOBC_TARGETS:
        vessel_name = target_col.replace("target_kobc_", "").replace("_next", "").capitalize()
        plt.figure(figsize=(12, 5))
        dates_plt = pd.to_datetime(predictions_df[DATE_COLUMN])
        plt.plot(dates_plt, predictions_df[f"actual_{target_col}"], label="Actual Observed (t+1)", color="black", linewidth=1.5)
        plt.plot(dates_plt, predictions_df[f"pred_persistence_{target_col}"], label="Persistence", linestyle="--", alpha=0.7, color="gray")
        plt.plot(dates_plt, predictions_df[f"pred_ridge_{target_col}"], label="Ridge (alpha=1)", alpha=0.8, color="blue")
        plt.plot(dates_plt, predictions_df[f"pred_xgboost_{target_col}"], label="XGBoost", alpha=0.8, color="green")
        plt.plot(dates_plt, predictions_df[f"pred_lstm_{target_col}"], label="LSTM", alpha=0.8, color="purple")
        plt.title(f"KOBC {vessel_name} Next-Trading-Day Forecast vs Actuals (Test Period)")
        plt.xlabel("Date")
        plt.ylabel("Freight Rate / Index")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / f"forecast_{target_col}.png", dpi=300)
        plt.close()

    # 2. MAE Comparison Chart
    plt.figure(figsize=(10, 5))
    sns.barplot(data=metrics_df, x="vessel", y="mae", hue="model", palette="muted")
    plt.title("Out-of-Sample MAE Comparison across KOBC Vessel Classes")
    plt.ylabel("MAE (points/rate)")
    plt.xlabel("Vessel Class")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "mae_comparison.png", dpi=300)
    plt.close()

    # 3. Directional Accuracy Comparison Chart
    plt.figure(figsize=(10, 5))
    sns.barplot(data=metrics_df, x="vessel", y="da_pct", hue="model", palette="muted")
    plt.title("Directional Accuracy (%) across KOBC Vessel Classes")
    plt.ylabel("Directional Accuracy (%)")
    plt.xlabel("Vessel Class")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "directional_accuracy_comparison.png", dpi=300)
    plt.close()

    # Model configuration metadata
    model_config = {
        "dataset": "data/features/freight_features_expanded.csv",
        "regime": "KOBC (2020+)",
        "total_usable_observations": n,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_date_range": [train_df[DATE_COLUMN].min().strftime("%Y-%m-%d"), train_df[DATE_COLUMN].max().strftime("%Y-%m-%d")],
        "val_date_range": [val_df[DATE_COLUMN].min().strftime("%Y-%m-%d"), val_df[DATE_COLUMN].max().strftime("%Y-%m-%d")],
        "test_date_range": [test_df[DATE_COLUMN].min().strftime("%Y-%m-%d"), test_df[DATE_COLUMN].max().strftime("%Y-%m-%d")],
        "feature_count": len(features),
        "features": features,
        "models_evaluated": ["Persistence", "Ridge", "XGBoost", "LSTM"],
        "targets": list(KOBC_TARGETS.keys()),
    }
    with open(output_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    print(f"Saved model config to: {output_dir / 'model_config.json'}")

    # Generate experiment report
    report_md = f"""# Expanded KOBC Freight Model Retraining Report

## 1. Executive Summary
This experiment retrained four forecasting architectures (**Persistence, Ridge Regression, XGBoost, PyTorch LSTM**) on the genuine 2020+ KOBC freight dataset from `data/features/freight_features_expanded.csv`.

- **Total Usable KOBC Rows:** {n} (from 2020-02-06 to 2026-08-31)
- **Train Set (70%):** {len(train_df)} rows ({train_df[DATE_COLUMN].min().strftime('%Y-%m-%d')} to {train_df[DATE_COLUMN].max().strftime('%Y-%m-%d')})
- **Validation Set (15%):** {len(val_df)} rows ({val_df[DATE_COLUMN].min().strftime('%Y-%m-%d')} to {val_df[DATE_COLUMN].max().strftime('%Y-%m-%d')})
- **Final Test Set (15%):** {len(test_df)} rows ({test_df[DATE_COLUMN].min().strftime('%Y-%m-%d')} to {test_df[DATE_COLUMN].max().strftime('%Y-%m-%d')})
- **Engineered Causal Features Used:** {len(features)}

---

## 2. Test Set Evaluation Metrics

| Vessel / Target | Model | MAE | RMSE | sMAPE (%) | R² | Directional Accuracy (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for _, row in metrics_df.iterrows():
        report_md += f"| **{row['vessel']}** | {row['model']} | {row['mae']:.2f} | {row['rmse']:.2f} | {row['smape']:.2f}% | {row['r2']:.4f} | {row['da_pct']:.2f}% |\n"

    report_md += f"""
---

## 3. Walk-Forward Cross-Validation (5 Folds Aggregate)

| Vessel | Model | Mean MAE ± Std | Mean RMSE ± Std | Mean sMAPE ± Std (%) | Mean DA ± Std (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    agg = fold_metrics_df.groupby(["vessel", "model"]).agg({
        "mae": ["mean", "std"],
        "rmse": ["mean", "std"],
        "smape": ["mean", "std"],
        "da_pct": ["mean", "std"],
    }).reset_index()

    for _, r in agg.iterrows():
        v = r["vessel"].values[0] if hasattr(r["vessel"], "values") else r["vessel"]
        m = r["model"].values[0] if hasattr(r["model"], "values") else r["model"]
        mae_m, mae_s = r[("mae", "mean")], r[("mae", "std")]
        rmse_m, rmse_s = r[("rmse", "mean")], r[("rmse", "std")]
        smape_m, smape_s = r[("smape", "mean")], r[("smape", "std")]
        da_m, da_s = r[("da_pct", "mean")], r[("da_pct", "std")]
        report_md += f"| **{v}** | {m} | {mae_m:.2f} ± {mae_s:.2f} | {rmse_m:.2f} ± {rmse_s:.2f} | {smape_m:.2f}% ± {smape_s:.2f}% | {da_m:.2f}% ± {da_s:.2f}% |\n"

    report_md += """
---

## 4. Model Selection & Champion Determination
Champion models per target are selected based on lowest MAE/sMAPE and superior directional accuracy across both holdout test and expanding walk-forward folds.
"""
    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved experiment report to: {output_dir / 'report.md'}")

    return metrics_df, fold_metrics_df


if __name__ == "__main__":
    run_expanded_experiment()
