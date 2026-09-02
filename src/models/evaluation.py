"""Model evaluation metrics, chronological holdout splitting, and benchmark execution."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.schemas import DATE_COLUMN, TARGET_COLUMNS
from src.models.baselines import PersistenceForecaster, MovingAverageForecaster
from src.models.ridge import RidgeForecaster


def compute_regression_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_current: Optional[Union[pd.Series, np.ndarray]] = None,
) -> Dict[str, float]:
    """Calculate comprehensive time-series forecasting metrics.

    Args:
        y_true: Actual ground truth values y_{t+1}.
        y_pred: Predicted values y_hat_{t+1}.
        y_current: Current freight level y_t (required for directional accuracy).

    Returns:
        Dict[str, float]: MAE, RMSE, MAPE, sMAPE, R2, and Directional Accuracy.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    # Filter out NaNs if any exist
    valid_mask = ~np.isnan(yt) & ~np.isnan(yp)
    yt = yt[valid_mask]
    yp = yp[valid_mask]

    if len(yt) == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "smape": np.nan, "r2": np.nan, "da_pct": np.nan}

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp))

    # MAPE (with safe handling of near-zero denominators)
    nonzero_mask = yt != 0
    if nonzero_mask.sum() > 0:
        mape = float(np.mean(np.abs((yt[nonzero_mask] - yp[nonzero_mask]) / yt[nonzero_mask])) * 100.0)
    else:
        mape = np.nan

    # sMAPE: 200 * |y - y_hat| / (|y| + |y_hat|)
    denom = np.abs(yt) + np.abs(yp)
    valid_denom = denom > 0
    if valid_denom.sum() > 0:
        smape = float(np.mean(2.0 * np.abs(yt[valid_denom] - yp[valid_denom]) / denom[valid_denom]) * 100.0)
    else:
        smape = np.nan

    # Directional Accuracy: comparing sign(y_{t+1} - y_t) with sign(y_hat_{t+1} - y_t)
    da_pct = np.nan
    if y_current is not None:
        yc = np.asarray(y_current, dtype=float)[valid_mask]
        actual_direction = np.sign(yt - yc)
        pred_direction = np.sign(yp - yc)
        correct_directions = (actual_direction == pred_direction)
        da_pct = float(np.mean(correct_directions) * 100.0)

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "smape": round(smape, 2),
        "r2": round(r2, 4),
        "da_pct": round(da_pct, 2) if not np.isnan(da_pct) else np.nan,
    }


def split_chronological_holdout(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    drop_initial_cold_start: int = 21,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform strict chronological train/test splitting (no shuffling, no lookahead).

    Args:
        df: Input feature DataFrame.
        train_ratio: Fraction of earliest observations for training.
        drop_initial_cold_start: Number of initial rows to drop due to lag cold-start.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df).
    """
    # Sort strictly by date ascending
    sorted_df = df.sort_values(by=DATE_COLUMN, ascending=True).reset_index(drop=True)

    # Exclude initial cold start rows where maximum lags (21) are unpopulated
    if drop_initial_cold_start > 0 and len(sorted_df) > drop_initial_cold_start:
        valid_df = sorted_df.iloc[drop_initial_cold_start:].copy()
    else:
        valid_df = sorted_df.copy()

    # Exclude the final row if its target is unobserved (active live forecast step)
    target_cols = [c for c in valid_df.columns if c.startswith("target_")]
    if target_cols:
        valid_df = valid_df[valid_df[target_cols[0]].notnull()].copy()

    valid_df = valid_df.reset_index(drop=True)
    n_total = len(valid_df)
    n_train = int(n_total * train_ratio)

    train_df = valid_df.iloc[:n_train].copy().reset_index(drop=True)
    test_df = valid_df.iloc[n_train:].copy().reset_index(drop=True)

    return train_df, test_df


def run_phase5_experiment(
    config_path: Union[str, Path] = "configs/models.yaml"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Execute complete Phase 5 baseline benchmark across all 4 freight targets.

    Evaluates:
    1. Naive Persistence (t -> t+1)
    2. Moving Averages (windows 3, 5, 10, 21)
    3. Ridge Regression (alphas 0.1, 1.0, 10.0, 100.0)

    Args:
        config_path: Path to models.yaml configuration file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict]: (metrics_df, predictions_df, metadata).
    """
    cfg_path = Path(config_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    feat_path = Path(cfg.get("data", {}).get("features_path", "data/features/freight_features.csv"))
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature dataset not found at: {feat_path.resolve()}")

    df_features = pd.read_csv(feat_path)
    df_features[DATE_COLUMN] = pd.to_datetime(df_features[DATE_COLUMN])

    train_ratio = float(cfg.get("data", {}).get("train_ratio", 0.80))
    cold_start = int(cfg.get("data", {}).get("drop_initial_cold_start", 21))

    # Perform strict chronological splitting
    train_df, test_df = split_chronological_holdout(
        df_features, train_ratio=train_ratio, drop_initial_cold_start=cold_start
    )

    targets_cfg = cfg.get("targets", {})
    ma_windows = cfg.get("baselines", {}).get("moving_average", {}).get("windows", [3, 5, 10, 21])
    ridge_alphas = cfg.get("baselines", {}).get("ridge", {}).get("alphas", [0.1, 1.0, 10.0, 100.0])

    all_metrics: List[Dict[str, Any]] = []
    pred_dict: Dict[str, Any] = {
        DATE_COLUMN: test_df[DATE_COLUMN].dt.strftime("%Y-%m-%d"),
    }

    # Feature columns for Ridge (exclude date, flags, and target columns)
    target_col_names = [info["target_col"] for info in targets_cfg.values()]
    exclude_cols = [DATE_COLUMN, "is_bdi_trading_day"] + target_col_names
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    for target_key, target_info in targets_cfg.items():
        target_col = target_info["target_col"]
        level_col = target_info["level_col"]

        y_train = train_df[target_col]
        y_test = test_df[target_col]
        y_current_test = test_df[level_col]

        pred_dict[f"actual_{target_key}"] = y_test.values
        pred_dict[f"level_{target_key}"] = y_current_test.values

        # -------------------------------------------------------------
        # Baseline 1: Naive Persistence
        # -------------------------------------------------------------
        persist_model = PersistenceForecaster()
        y_pred_persist = persist_model.predict(y_current_test)
        pred_dict[f"pred_{target_key}_persistence"] = y_pred_persist

        m_persist = compute_regression_metrics(y_test, y_pred_persist, y_current_test)
        all_metrics.append({
            "target": target_key,
            "model": "Persistence",
            "hyperparameters": "None",
            **m_persist,
        })

        # -------------------------------------------------------------
        # Baseline 2: Moving Averages
        # -------------------------------------------------------------
        for w in ma_windows:
            ma_model = MovingAverageForecaster(window=w)
            # Combine train and test historical series to get causal rolling window across the boundary
            full_series = pd.concat([train_df[level_col], test_df[level_col]], ignore_index=True)
            full_ma_pred = ma_model.predict(full_series)
            test_ma_pred = full_ma_pred.iloc[len(train_df):].values

            pred_dict[f"pred_{target_key}_ma_{w}"] = test_ma_pred
            m_ma = compute_regression_metrics(y_test, test_ma_pred, y_current_test)
            all_metrics.append({
                "target": target_key,
                "model": f"MovingAverage_{w}",
                "hyperparameters": f"window={w}",
                **m_ma,
            })

        # -------------------------------------------------------------
        # Baseline 3: Ridge Regression
        # -------------------------------------------------------------
        for alpha in ridge_alphas:
            ridge_model = RidgeForecaster(alpha=alpha, scale_features=True)
            ridge_model.fit(X_train, y_train)
            y_pred_ridge = ridge_model.predict(X_test)

            pred_dict[f"pred_{target_key}_ridge_alpha_{alpha}"] = y_pred_ridge
            m_ridge = compute_regression_metrics(y_test, y_pred_ridge, y_current_test)
            all_metrics.append({
                "target": target_key,
                "model": f"Ridge_a{alpha}",
                "hyperparameters": f"alpha={alpha}",
                **m_ridge,
            })

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.DataFrame(pred_dict)

    # Save experiment outputs
    out_dir = Path(cfg.get("output", {}).get("experiment_dir", "experiments/phase5"))
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.csv"
    predictions_path = out_dir / "predictions.csv"
    config_saved_path = out_dir / "configuration.yaml"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    with open(config_saved_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # Generate residual & actual vs predicted plots
    generate_baseline_plots(test_df, predictions_df, targets_cfg, out_dir / "figures")

    metadata = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_date_min": train_df[DATE_COLUMN].min().strftime("%Y-%m-%d"),
        "train_date_max": train_df[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        "test_date_min": test_df[DATE_COLUMN].min().strftime("%Y-%m-%d"),
        "test_date_max": test_df[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        "total_models_evaluated": len(metrics_df),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
    }

    return metrics_df, predictions_df, metadata


def generate_baseline_plots(
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    targets_cfg: Dict[str, Any],
    fig_dir: Path,
):
    """Generate visual diagnostics for baseline forecasts."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    dates = pd.to_datetime(pred_df[DATE_COLUMN])

    # 1. Actual vs Predicted for Best Baseline and Persistence
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    for i, (key, _) in enumerate(targets_cfg.items()):
        actual = pred_df[f"actual_{key}"]
        persist = pred_df[f"pred_{key}_persistence"]
        ridge_1 = pred_df.get(f"pred_{key}_ridge_alpha_1.0", persist)

        axes[i].plot(dates, actual, label="Actual Ground Truth", color="black", lw=1.5)
        axes[i].plot(dates, persist, label="Naive Persistence", color="#1f77b4", linestyle="--", alpha=0.8)
        axes[i].plot(dates, ridge_1, label="Ridge (alpha=1.0)", color="#d62728", lw=1.2)
        axes[i].set_title(f"{key.upper()} 1-Step-Ahead Forecasts vs Actual (Test Period)", fontweight="bold")
        axes[i].set_ylabel("Index Level")
        axes[i].legend(loc="upper left")

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    fig.savefig(fig_dir / "01_actual_vs_predicted_test.png", dpi=200)
    plt.close(fig)

    # 2. Residual Distribution & Error Time Series
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    for i, (key, _) in enumerate(targets_cfg.items()):
        actual = pred_df[f"actual_{key}"]
        persist = pred_df[f"pred_{key}_persistence"]
        res_persist = actual - persist

        # Residual Time Series
        axes[i, 0].plot(dates, res_persist, color="#1f77b4", lw=1.0)
        axes[i, 0].axhline(0, color="black", linestyle="--", lw=0.8)
        axes[i, 0].set_title(f"{key.upper()} Persistence Residual Over Time (e_t = y_{{t+1}} - y_t)", fontweight="bold")
        axes[i, 0].set_ylabel("Error (Index Units)")

        # Residual Distribution
        sns.histplot(res_persist, kde=True, ax=axes[i, 1], color="#1f77b4", bins=30)
        axes[i, 1].set_title(f"{key.upper()} Residual Distribution", fontweight="bold")
        axes[i, 1].set_xlabel("Prediction Error")

    plt.tight_layout()
    fig.savefig(fig_dir / "02_residuals_diagnostics.png", dpi=200)
    plt.close(fig)


def run_phase6_xgboost_experiment(
    config_path: Union[str, Path] = "configs/models.yaml"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Execute complete Phase 6 XGBoost benchmark across all 4 freight targets.

    Evaluates:
    - XGBoost models for HSI, SI, PI, CI
    - Direct performance comparison against Phase 5 baselines (Persistence, Ridge)
    - Feature importance analysis and model persistence

    Args:
        config_path: Path to models.yaml.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
            (metrics_df, predictions_df, feature_importance_df, metadata).
    """
    from src.models.xgboost_model import XGBoostForecaster

    cfg_path = Path(config_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    feat_path = Path(cfg.get("data", {}).get("features_path", "data/features/freight_features.csv"))
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature dataset not found at: {feat_path.resolve()}")

    df_features = pd.read_csv(feat_path)
    df_features[DATE_COLUMN] = pd.to_datetime(df_features[DATE_COLUMN])

    train_ratio = float(cfg.get("data", {}).get("train_ratio", 0.80))
    cold_start = int(cfg.get("data", {}).get("drop_initial_cold_start", 21))

    # Strict chronological split (exact same split as Phase 5)
    train_df, test_df = split_chronological_holdout(
        df_features, train_ratio=train_ratio, drop_initial_cold_start=cold_start
    )

    targets_cfg = cfg.get("targets", {})
    xgb_params = cfg.get("xgboost", {})

    target_col_names = [info["target_col"] for info in targets_cfg.values()]
    exclude_cols = [DATE_COLUMN, "is_bdi_trading_day"] + target_col_names
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    out_dir = Path(cfg.get("output", {}).get("phase6_experiment_dir", "experiments/phase6"))
    models_dir = out_dir / "models"
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict[str, Any]] = []
    all_importances: List[pd.DataFrame] = []
    pred_dict: Dict[str, Any] = {
        DATE_COLUMN: test_df[DATE_COLUMN].dt.strftime("%Y-%m-%d"),
    }

    trained_models: Dict[str, XGBoostForecaster] = {}

    for target_key, target_info in targets_cfg.items():
        target_col = target_info["target_col"]
        level_col = target_info["level_col"]

        y_train = train_df[target_col]
        y_test = test_df[target_col]
        y_current_test = test_df[level_col]

        pred_dict[f"actual_{target_key}"] = y_test.values
        pred_dict[f"level_{target_key}"] = y_current_test.values

        # 1. Baseline 1: Naive Persistence for direct in-table comparison
        persist_model = PersistenceForecaster()
        y_pred_persist = persist_model.predict(y_current_test)
        pred_dict[f"pred_{target_key}_persistence"] = y_pred_persist
        m_persist = compute_regression_metrics(y_test, y_pred_persist, y_current_test)
        all_metrics.append({
            "target": target_key,
            "model": "Persistence",
            **m_persist,
        })

        # 2. Baseline 2: Ridge (alpha=1.0) for direct benchmark comparison
        ridge_model = RidgeForecaster(alpha=1.0, scale_features=True)
        ridge_model.fit(X_train, y_train)
        y_pred_ridge = ridge_model.predict(X_test)
        pred_dict[f"pred_{target_key}_ridge"] = y_pred_ridge
        m_ridge = compute_regression_metrics(y_test, y_pred_ridge, y_current_test)
        all_metrics.append({
            "target": target_key,
            "model": "Ridge_a1.0",
            **m_ridge,
        })

        # 3. XGBoost Model Training & Resumability Check
        model_artifact_path = models_dir / f"xgboost_{target_key}.json"
        xgb_model = XGBoostForecaster(**xgb_params)

        xgb_model.fit(X_train, y_train)
        xgb_model.save_model(model_artifact_path)
        trained_models[target_key] = xgb_model

        y_pred_xgb = xgb_model.predict(X_test)
        pred_dict[f"pred_{target_key}_xgboost"] = y_pred_xgb
        m_xgb = compute_regression_metrics(y_test, y_pred_xgb, y_current_test)
        all_metrics.append({
            "target": target_key,
            "model": "XGBoost",
            **m_xgb,
        })

        # 4. Feature Importance Extraction
        imp_df = xgb_model.get_feature_importances(importance_type="gain", top_n=20)
        imp_df["target"] = target_key
        all_importances.append(imp_df)

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.DataFrame(pred_dict)
    feature_importance_df = pd.concat(all_importances, ignore_index=True)

    # Save outputs
    metrics_path = out_dir / "metrics.csv"
    predictions_path = out_dir / "predictions.csv"
    imp_path = out_dir / "feature_importance.csv"
    config_saved_path = out_dir / "configuration.yaml"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    feature_importance_df.to_csv(imp_path, index=False)
    with open(config_saved_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # Generate XGBoost Diagnostic Figures
    generate_xgboost_plots(test_df, predictions_df, feature_importance_df, targets_cfg, figures_dir)

    metadata = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_date_min": train_df[DATE_COLUMN].min().strftime("%Y-%m-%d"),
        "train_date_max": train_df[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        "test_date_min": test_df[DATE_COLUMN].min().strftime("%Y-%m-%d"),
        "test_date_max": test_df[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "feature_importance_path": str(imp_path),
        "models_dir": str(models_dir),
    }

    return metrics_df, predictions_df, feature_importance_df, metadata


def generate_xgboost_plots(
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    imp_df: pd.DataFrame,
    targets_cfg: Dict[str, Any],
    fig_dir: Path,
):
    """Generate diagnostic visual plots for XGBoost predictions and feature importances."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    dates = pd.to_datetime(pred_df[DATE_COLUMN])

    # 1. Actual vs Ridge vs XGBoost Forecasts
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    for i, (key, _) in enumerate(targets_cfg.items()):
        actual = pred_df[f"actual_{key}"]
        ridge = pred_df[f"pred_{key}_ridge"]
        xgb_pred = pred_df[f"pred_{key}_xgboost"]

        axes[i].plot(dates, actual, label="Actual Ground Truth", color="black", lw=1.5)
        axes[i].plot(dates, ridge, label="Ridge (alpha=1.0)", color="#1f77b4", linestyle="--", alpha=0.8)
        axes[i].plot(dates, xgb_pred, label="XGBoost", color="#2ca02c", lw=1.3)
        axes[i].set_title(f"{key.upper()} Test Horizon Forecasts: Actual vs Ridge vs XGBoost", fontweight="bold")
        axes[i].set_ylabel("Index Level")
        axes[i].legend(loc="upper left")

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    fig.savefig(fig_dir / "01_actual_vs_predicted_xgb_test.png", dpi=200)
    plt.close(fig)

    # 2. Residual Distribution & Error Over Time
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    for i, (key, _) in enumerate(targets_cfg.items()):
        actual = pred_df[f"actual_{key}"]
        xgb_pred = pred_df[f"pred_{key}_xgboost"]
        res_xgb = actual - xgb_pred

        # Residual Time Series
        axes[i, 0].plot(dates, res_xgb, color="#2ca02c", lw=1.0)
        axes[i, 0].axhline(0, color="black", linestyle="--", lw=0.8)
        axes[i, 0].set_title(f"{key.upper()} XGBoost Residual Over Time (e_t)", fontweight="bold")
        axes[i, 0].set_ylabel("Error (Index Units)")

        # Residual Distribution
        sns.histplot(res_xgb, kde=True, ax=axes[i, 1], color="#2ca02c", bins=30)
        axes[i, 1].set_title(f"{key.upper()} XGBoost Residual Distribution", fontweight="bold")
        axes[i, 1].set_xlabel("Prediction Error")

    plt.tight_layout()
    fig.savefig(fig_dir / "02_residuals_diagnostics_xgb.png", dpi=200)
    plt.close(fig)

    # 3. Top 10 Feature Importances (Gain) for Each Vessel Target
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()
    for i, (key, _) in enumerate(targets_cfg.items()):
        sub_imp = imp_df[imp_df["target"] == key].head(10).sort_values(by="importance", ascending=True)
        axes[i].barh(sub_imp["feature"], sub_imp["importance"], color="#2ca02c")
        axes[i].set_title(f"Top 10 Features for {key.upper()} (Gain Importance)", fontweight="bold")
        axes[i].set_xlabel("Gain")

    plt.tight_layout()
    fig.savefig(fig_dir / "03_feature_importance_top15.png", dpi=200)
    plt.close(fig)


def get_lstm_feature_subset(df_columns: List[str]) -> List[str]:
    """Select a controlled, parsimonious feature subset for LSTM training."""
    selected = []
    # 1. Base levels
    selected.extend([c for c in df_columns if c.endswith("_level")])
    # 2. Key autoregressive lags (1, 2, 3, 5)
    for lag in [1, 2, 3, 5]:
        selected.extend([c for c in df_columns if f"_lag_{lag}" in c and not c.startswith(("cross_", "wti_", "brent_", "usd_", "gpr_", "wind_", "precip_", "pressure_"))])
    # 3. Cross-vessel lags
    selected.extend([c for c in df_columns if c.startswith("cross_")])
    # 4. Short-term differences and returns
    for diff in [1]:
        selected.extend([c for c in df_columns if f"_diff_{diff}" in c and not c.startswith(("gpr_", "pressure_"))])
        selected.extend([c for c in df_columns if f"_pct_change_{diff}" in c and not c.startswith(("wti_", "brent_", "usd_", "gpr_"))])
    # 5. Short rolling statistics & volatility
    selected.extend([c for c in df_columns if "_roll_mean_7" in c or "_roll_std_7" in c or "_return_vol_7" in c])
    # 6. Selected exogenous macro & GPR
    selected.extend([c for c in df_columns if c in [
        "wti_usd_bbl_lag_1", "brent_usd_bbl_lag_1", "usd_inr_lag_1", "gpr_lag_1", "gpr_spike_ratio_ma30"
    ]])

    # Deduplicate while preserving order
    seen = set()
    result = []
    for f in selected:
        if f in df_columns and f not in seen:
            seen.add(f)
            result.append(f)
    return result


def run_phase7_lstm_experiment(
    config_path: Union[str, Path] = "configs/models.yaml"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Execute complete Phase 7 LSTM benchmark across all 4 freight targets.

    Evaluates:
    - PyTorch LSTM models for HSI, SI, PI, CI with 21-day sliding lookback
    - Direct performance comparison against Persistence, Ridge, and XGBoost
    - Diagnostics and model persistence

    Args:
        config_path: Path to models.yaml.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict]: (metrics_df, predictions_df, metadata).
    """
    from src.models.lstm_model import LSTMForecaster

    cfg_path = Path(config_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    feat_path = Path(cfg.get("data", {}).get("features_path", "data/features/freight_features.csv"))
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature dataset not found at: {feat_path.resolve()}")

    df_features = pd.read_csv(feat_path)
    df_features[DATE_COLUMN] = pd.to_datetime(df_features[DATE_COLUMN])

    train_ratio = float(cfg.get("data", {}).get("train_ratio", 0.80))
    cold_start = int(cfg.get("data", {}).get("drop_initial_cold_start", 21))

    # Strict chronological holdout split (exact same as Phase 5 and Phase 6)
    train_df, test_df = split_chronological_holdout(
        df_features, train_ratio=train_ratio, drop_initial_cold_start=cold_start
    )

    targets_cfg = cfg.get("targets", {})
    lstm_cfg = cfg.get("lstm", {})

    # Full combined dataset for continuous lookback sequences across the train/test boundary
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    test_start_idx = len(train_df)

    # Controlled feature subset
    feature_cols = get_lstm_feature_subset(list(df_features.columns))

    out_dir = Path(cfg.get("output", {}).get("phase7_experiment_dir", "experiments/phase7"))
    models_dir = out_dir / "models"
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict[str, Any]] = []
    training_histories: Dict[str, Dict[str, List[float]]] = {}
    pred_dict: Dict[str, Any] = {
        DATE_COLUMN: test_df[DATE_COLUMN].dt.strftime("%Y-%m-%d"),
    }

    # Load Phase 5 and Phase 6 predictions if available to compile comprehensive predictions table
    p5_pred_path = Path("experiments/phase5/predictions.csv")
    p6_pred_path = Path("experiments/phase6/predictions.csv")
    p5_preds = pd.read_csv(p5_pred_path) if p5_pred_path.exists() else None
    p6_preds = pd.read_csv(p6_pred_path) if p6_pred_path.exists() else None

    # Load Phase 5 & 6 metrics for combined benchmark table
    p5_metrics_path = Path("experiments/phase5/metrics.csv")
    p6_metrics_path = Path("experiments/phase6/metrics.csv")
    existing_metrics_df = None
    if p6_metrics_path.exists():
        existing_metrics_df = pd.read_csv(p6_metrics_path)
    elif p5_metrics_path.exists():
        existing_metrics_df = pd.read_csv(p5_metrics_path)

    for target_key, target_info in targets_cfg.items():
        target_col = target_info["target_col"]
        level_col = target_info["level_col"]

        y_train = train_df[target_col]
        y_test = test_df[target_col]
        y_current_test = test_df[level_col]

        pred_dict[f"actual_{target_key}"] = y_test.values
        pred_dict[f"level_{target_key}"] = y_current_test.values

        # Include Persistence, Ridge, and XGBoost predictions if available
        if p5_preds is not None and f"pred_{target_key}_persistence" in p5_preds.columns:
            pred_dict[f"pred_{target_key}_persistence"] = p5_preds[f"pred_{target_key}_persistence"].values
        if p5_preds is not None and f"pred_{target_key}_ridge_alpha_1.0" in p5_preds.columns:
            pred_dict[f"pred_{target_key}_ridge"] = p5_preds[f"pred_{target_key}_ridge_alpha_1.0"].values
        if p6_preds is not None and f"pred_{target_key}_xgboost" in p6_preds.columns:
            pred_dict[f"pred_{target_key}_xgboost"] = p6_preds[f"pred_{target_key}_xgboost"].values

        # 1. Baseline 1: Naive Persistence metric
        persist_model = PersistenceForecaster()
        y_pred_persist = persist_model.predict(y_current_test)
        m_persist = compute_regression_metrics(y_test, y_pred_persist, y_current_test)
        all_metrics.append({"target": target_key, "model": "Persistence", **m_persist})

        # 2. Baseline 2: Ridge (alpha=1.0) metric
        ridge_model = RidgeForecaster(alpha=1.0, scale_features=True)
        ridge_model.fit(train_df[[c for c in df_features.columns if c not in [DATE_COLUMN, "is_bdi_trading_day"] + [info["target_col"] for info in targets_cfg.values()]]], y_train)
        y_pred_ridge = ridge_model.predict(test_df[[c for c in df_features.columns if c not in [DATE_COLUMN, "is_bdi_trading_day"] + [info["target_col"] for info in targets_cfg.values()]]])
        m_ridge = compute_regression_metrics(y_test, y_pred_ridge, y_current_test)
        all_metrics.append({"target": target_key, "model": "Ridge_a1.0", **m_ridge})

        # 3. Model 3: XGBoost metric if available from Phase 6
        if existing_metrics_df is not None:
            xgb_row = existing_metrics_df[(existing_metrics_df["target"] == target_key) & (existing_metrics_df["model"] == "XGBoost")]
            if len(xgb_row) > 0:
                all_metrics.append(xgb_row.iloc[0].to_dict())

        # 4. Model 4: Train LSTM Model
        lstm_model = LSTMForecaster(
            lookback=lstm_cfg.get("lookback", 21),
            hidden_size=lstm_cfg.get("hidden_size", 64),
            dense_units=lstm_cfg.get("dense_units", 32),
            num_layers=lstm_cfg.get("num_layers", 1),
            dropout=lstm_cfg.get("dropout", 0.15),
            learning_rate=lstm_cfg.get("learning_rate", 0.001),
            weight_decay=lstm_cfg.get("weight_decay", 1e-4),
            batch_size=lstm_cfg.get("batch_size", 32),
            max_epochs=lstm_cfg.get("max_epochs", 60),
            early_stopping_patience=lstm_cfg.get("early_stopping_patience", 10),
            random_seed=lstm_cfg.get("random_seed", 42),
        )

        lstm_model.fit(train_df[feature_cols], y_train, val_ratio=lstm_cfg.get("val_ratio", 0.15), verbose=False)
        
        # Save model checkpoint
        model_artifact_path = models_dir / f"lstm_{target_key}.pt"
        lstm_model.save_model(model_artifact_path)
        training_histories[target_key] = lstm_model.history

        # Predict across boundary
        y_pred_lstm = lstm_model.predict_test_boundary(full_df[feature_cols], test_start_idx)
        pred_dict[f"pred_{target_key}_lstm"] = y_pred_lstm

        m_lstm = compute_regression_metrics(y_test, y_pred_lstm, y_current_test)
        all_metrics.append({"target": target_key, "model": "LSTM", **m_lstm})

    metrics_df = pd.DataFrame(all_metrics).drop_duplicates(subset=["target", "model"]).reset_index(drop=True)
    predictions_df = pd.DataFrame(pred_dict)

    # Save outputs
    metrics_path = out_dir / "metrics.csv"
    predictions_path = out_dir / "predictions.csv"
    config_saved_path = out_dir / "configuration.yaml"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    with open(config_saved_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # Generate LSTM Diagnostic Figures
    generate_lstm_plots(test_df, predictions_df, training_histories, targets_cfg, figures_dir)

    metadata = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "lookback": lstm_cfg.get("lookback", 21),
        "features_count": len(feature_cols),
        "features_used": feature_cols,
        "train_date_min": train_df[DATE_COLUMN].min().strftime("%Y-%m-%d"),
        "train_date_max": train_df[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        "test_date_min": test_df[DATE_COLUMN].min().strftime("%Y-%m-%d"),
        "test_date_max": test_df[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "models_dir": str(models_dir),
    }

    return metrics_df, predictions_df, metadata


def generate_lstm_plots(
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    histories: Dict[str, Dict[str, List[float]]],
    targets_cfg: Dict[str, Any],
    fig_dir: Path,
):
    """Generate visual diagnostics for LSTM predictions, residuals, and training histories."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    dates = pd.to_datetime(pred_df[DATE_COLUMN])

    # 1. Multi-Model Forecast Comparison (Actual vs Ridge vs XGBoost vs LSTM)
    fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)
    for i, (key, _) in enumerate(targets_cfg.items()):
        actual = pred_df[f"actual_{key}"]
        ridge = pred_df.get(f"pred_{key}_ridge", pred_df[f"actual_{key}"])
        xgb_p = pred_df.get(f"pred_{key}_xgboost", pred_df[f"actual_{key}"])
        lstm_p = pred_df[f"pred_{key}_lstm"]

        axes[i].plot(dates, actual, label="Actual Ground Truth", color="black", lw=1.6)
        axes[i].plot(dates, ridge, label="Ridge (alpha=1.0)", color="#1f77b4", linestyle="--", alpha=0.8)
        axes[i].plot(dates, xgb_p, label="XGBoost", color="#2ca02c", linestyle=":", alpha=0.8)
        axes[i].plot(dates, lstm_p, label="LSTM", color="#9467bd", lw=1.3)
        axes[i].set_title(f"{key.upper()} Multi-Model Comparison (Test Horizon: 2018-03 to 2019-07)", fontweight="bold")
        axes[i].set_ylabel("Index Level")
        axes[i].legend(loc="upper left")

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    fig.savefig(fig_dir / "01_actual_vs_all_models_test.png", dpi=200)
    plt.close(fig)

    # 2. LSTM Residual Diagnostics
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    for i, (key, _) in enumerate(targets_cfg.items()):
        actual = pred_df[f"actual_{key}"]
        lstm_p = pred_df[f"pred_{key}_lstm"]
        res_lstm = actual - lstm_p

        # Residual Time Series
        axes[i, 0].plot(dates, res_lstm, color="#9467bd", lw=1.0)
        axes[i, 0].axhline(0, color="black", linestyle="--", lw=0.8)
        axes[i, 0].set_title(f"{key.upper()} LSTM Residual Over Time (e_t)", fontweight="bold")
        axes[i, 0].set_ylabel("Error (Index Units)")

        # Residual Distribution
        sns.histplot(res_lstm, kde=True, ax=axes[i, 1], color="#9467bd", bins=30)
        axes[i, 1].set_title(f"{key.upper()} LSTM Residual Distribution", fontweight="bold")
        axes[i, 1].set_xlabel("Prediction Error")

    plt.tight_layout()
    fig.savefig(fig_dir / "02_lstm_residuals_diagnostics.png", dpi=200)
    plt.close(fig)

    # 3. Training & Validation Loss Curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for i, (key, hist) in enumerate(histories.items()):
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[i].plot(epochs, hist["train_loss"], label="Train Loss (MSE)", color="#1f77b4", lw=1.5)
        axes[i].plot(epochs, hist["val_loss"], label="Val Loss (MSE)", color="#d62728", lw=1.5)
        axes[i].set_title(f"{key.upper()} LSTM Training & Validation Loss", fontweight="bold")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("MSE Loss")
        axes[i].legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(fig_dir / "03_lstm_training_history.png", dpi=200)
    plt.close(fig)


def generate_walk_forward_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    test_window_size: int = 200,
    drop_initial_cold_start: int = 21,
) -> List[Dict[str, Any]]:
    """Generate expanding-window chronological walk-forward folds.

    Args:
        df: Full feature dataset.
        n_folds: Number of evaluation folds.
        test_window_size: Number of trading sessions per test fold.
        drop_initial_cold_start: Initial cold-start rows to exclude.

    Returns:
        List[Dict[str, Any]]: List of fold dictionaries containing train_df, test_df, and metadata.
    """
    sorted_df = df.sort_values(by=DATE_COLUMN, ascending=True).reset_index(drop=True)
    if drop_initial_cold_start > 0 and len(sorted_df) > drop_initial_cold_start:
        valid_df = sorted_df.iloc[drop_initial_cold_start:].copy()
    else:
        valid_df = sorted_df.copy()

    # Exclude unobserved final live target
    target_cols = [c for c in valid_df.columns if c.startswith("target_")]
    if target_cols:
        valid_df = valid_df[valid_df[target_cols[0]].notnull()].copy()

    valid_df = valid_df.reset_index(drop=True)
    N = len(valid_df)
    total_test_span = n_folds * test_window_size
    min_train_size = N - total_test_span

    if min_train_size < 100:
        raise ValueError(
            f"Insufficient observations ({N}) for {n_folds} folds of size {test_window_size}. "
            f"Minimum training size would be {min_train_size}."
        )

    folds = []
    for k in range(n_folds):
        train_end = min_train_size + k * test_window_size
        test_end = train_end + test_window_size

        train_slice = valid_df.iloc[:train_end].copy().reset_index(drop=True)
        test_slice = valid_df.iloc[train_end:test_end].copy().reset_index(drop=True)

        folds.append({
            "fold": k + 1,
            "train_df": train_slice,
            "test_df": test_slice,
            "train_start": train_slice[DATE_COLUMN].min(),
            "train_end": train_slice[DATE_COLUMN].max(),
            "test_start": test_slice[DATE_COLUMN].min(),
            "test_end": test_slice[DATE_COLUMN].max(),
            "train_size": len(train_slice),
            "test_size": len(test_slice),
            "train_end_idx": train_end,
            "full_df": valid_df,
        })

    return folds


def run_phase8_walk_forward_experiment(
    config_path: Union[str, Path] = "configs/models.yaml"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Execute complete Phase 8 Expanding-Window Walk-Forward Evaluation across all 4 freight targets.

    Evaluates across N expanding folds:
    1. Naive Persistence
    2. Ridge Regression (alpha=1.0)
    3. XGBoost
    4. LSTM (Lookback=21)
    5. Simple Equal-Weighted Ensemble (Ridge + XGBoost)

    Args:
        config_path: Path to models.yaml.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
            (fold_metrics_df, aggregate_metrics_df, predictions_df, metadata).
    """
    from src.models.xgboost_model import XGBoostForecaster
    from src.models.lstm_model import LSTMForecaster

    cfg_path = Path(config_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    feat_path = Path(cfg.get("data", {}).get("features_path", "data/features/freight_features.csv"))
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature dataset not found at: {feat_path.resolve()}")

    df_features = pd.read_csv(feat_path)
    df_features[DATE_COLUMN] = pd.to_datetime(df_features[DATE_COLUMN])

    cold_start = int(cfg.get("data", {}).get("drop_initial_cold_start", 21))
    wf_cfg = cfg.get("walk_forward", {})
    n_folds = int(wf_cfg.get("n_folds", 5))
    test_window_size = int(wf_cfg.get("test_window_size", 200))
    include_lstm = bool(wf_cfg.get("include_lstm", True))
    eval_ensemble = bool(wf_cfg.get("evaluate_ensemble", True))

    targets_cfg = cfg.get("targets", {})
    xgb_params = cfg.get("xgboost", {})
    lstm_params = cfg.get("lstm", {})

    target_col_names = [info["target_col"] for info in targets_cfg.values()]
    exclude_cols = [DATE_COLUMN, "is_bdi_trading_day"] + target_col_names
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]
    lstm_features = get_lstm_feature_subset(list(df_features.columns))

    out_dir = Path(cfg.get("output", {}).get("phase8_experiment_dir", "experiments/phase8"))
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Expanding Folds
    folds = generate_walk_forward_folds(
        df_features, n_folds=n_folds, test_window_size=test_window_size, drop_initial_cold_start=cold_start
    )

    all_fold_metrics: List[Dict[str, Any]] = []
    all_fold_preds: List[pd.DataFrame] = []

    for fold_info in folds:
        fold_num = fold_info["fold"]
        train_df = fold_info["train_df"]
        test_df = fold_info["test_df"]
        full_df = fold_info["full_df"]
        test_start_idx = fold_info["train_end_idx"]

        pred_fold_dict: Dict[str, Any] = {
            "fold": fold_num,
            DATE_COLUMN: test_df[DATE_COLUMN].dt.strftime("%Y-%m-%d"),
        }

        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]

        for target_key, target_info in targets_cfg.items():
            target_col = target_info["target_col"]
            level_col = target_info["level_col"]

            y_train = train_df[target_col]
            y_test = test_df[target_col]
            y_current_test = test_df[level_col]

            pred_fold_dict[f"actual_{target_key}"] = y_test.values
            pred_fold_dict[f"level_{target_key}"] = y_current_test.values

            # Model 1: Persistence
            persist_model = PersistenceForecaster()
            y_pred_persist = persist_model.predict(y_current_test)
            pred_fold_dict[f"pred_{target_key}_persistence"] = y_pred_persist
            m_persist = compute_regression_metrics(y_test, y_pred_persist, y_current_test)
            all_fold_metrics.append({"fold": fold_num, "target": target_key, "model": "Persistence", **m_persist})

            # Model 2: Ridge (alpha=1.0)
            ridge_model = RidgeForecaster(alpha=1.0, scale_features=True)
            ridge_model.fit(X_train, y_train)
            y_pred_ridge = ridge_model.predict(X_test)
            pred_fold_dict[f"pred_{target_key}_ridge"] = y_pred_ridge
            m_ridge = compute_regression_metrics(y_test, y_pred_ridge, y_current_test)
            all_fold_metrics.append({"fold": fold_num, "target": target_key, "model": "Ridge_a1.0", **m_ridge})

            # Model 3: XGBoost
            xgb_model = XGBoostForecaster(**xgb_params)
            xgb_model.fit(X_train, y_train)
            y_pred_xgb = xgb_model.predict(X_test)
            pred_fold_dict[f"pred_{target_key}_xgboost"] = y_pred_xgb
            m_xgb = compute_regression_metrics(y_test, y_pred_xgb, y_current_test)
            all_fold_metrics.append({"fold": fold_num, "target": target_key, "model": "XGBoost", **m_xgb})

            # Model 4: LSTM (optional / controlled)
            y_pred_lstm = None
            if include_lstm:
                lstm_model = LSTMForecaster(
                    lookback=lstm_params.get("lookback", 21),
                    hidden_size=lstm_params.get("hidden_size", 64),
                    dense_units=lstm_params.get("dense_units", 32),
                    num_layers=lstm_params.get("num_layers", 1),
                    dropout=lstm_params.get("dropout", 0.15),
                    learning_rate=lstm_params.get("learning_rate", 0.001),
                    weight_decay=lstm_params.get("weight_decay", 1e-4),
                    batch_size=lstm_params.get("batch_size", 32),
                    max_epochs=lstm_params.get("max_epochs", 40),
                    early_stopping_patience=lstm_params.get("early_stopping_patience", 8),
                    random_seed=lstm_params.get("random_seed", 42) + fold_num,
                )
                lstm_model.fit(train_df[lstm_features], y_train, val_ratio=0.15, verbose=False)
                y_pred_lstm = lstm_model.predict_test_boundary(
                    full_df[lstm_features],
                    test_start_idx=test_start_idx,
                    test_end_idx=test_start_idx + len(test_df),
                )
                pred_fold_dict[f"pred_{target_key}_lstm"] = y_pred_lstm
                m_lstm = compute_regression_metrics(y_test, y_pred_lstm, y_current_test)
                all_fold_metrics.append({"fold": fold_num, "target": target_key, "model": "LSTM", **m_lstm})

            # Model 5: Simple Equal-Weighted Ensemble (Ridge + XGBoost)
            if eval_ensemble:
                y_pred_ens = 0.5 * y_pred_ridge + 0.5 * y_pred_xgb
                pred_fold_dict[f"pred_{target_key}_ensemble"] = y_pred_ens
                m_ens = compute_regression_metrics(y_test, y_pred_ens, y_current_test)
                all_fold_metrics.append({"fold": fold_num, "target": target_key, "model": "Ensemble_Ridge_XGB", **m_ens})

        all_fold_preds.append(pd.DataFrame(pred_fold_dict))

    fold_metrics_df = pd.DataFrame(all_fold_metrics)
    predictions_df = pd.concat(all_fold_preds, ignore_index=True)

    # 2. Compute Aggregate Metrics Across All Folds
    agg_list = []
    for (tgt, mdl), group in fold_metrics_df.groupby(["target", "model"]):
        agg_list.append({
            "target": tgt,
            "model": mdl,
            "mean_mae": round(float(group["mae"].mean()), 2),
            "median_mae": round(float(group["mae"].median()), 2),
            "std_mae": round(float(group["mae"].std()), 2),
            "mean_rmse": round(float(group["rmse"].mean()), 2),
            "mean_smape": round(float(group["smape"].mean()), 2),
            "mean_r2": round(float(group["r2"].mean()), 4),
            "mean_da_pct": round(float(group["da_pct"].mean()), 2),
            "n_folds": len(group),
        })

    aggregate_metrics_df = pd.DataFrame(agg_list).sort_values(by=["target", "mean_mae"]).reset_index(drop=True)

    # 3. Save Experiment Artifacts
    fold_metrics_path = out_dir / "fold_metrics.csv"
    agg_metrics_path = out_dir / "aggregate_metrics.csv"
    predictions_path = out_dir / "predictions.csv"
    config_saved_path = out_dir / "configuration.yaml"

    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    aggregate_metrics_df.to_csv(agg_metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    with open(config_saved_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # 4. Generate Diagnostic Figures
    generate_walk_forward_plots(fold_metrics_df, aggregate_metrics_df, predictions_df, targets_cfg, figures_dir)

    metadata = {
        "n_folds": n_folds,
        "test_window_size": test_window_size,
        "total_test_eval_points": len(predictions_df),
        "fold_definitions": [
            {
                "fold": f["fold"],
                "train_dates": f"{f['train_start'].strftime('%Y-%m-%d')} to {f['train_end'].strftime('%Y-%m-%d')}",
                "train_rows": f["train_size"],
                "test_dates": f"{f['test_start'].strftime('%Y-%m-%d')} to {f['test_end'].strftime('%Y-%m-%d')}",
                "test_rows": f["test_size"],
            }
            for f in folds
        ],
        "fold_metrics_path": str(fold_metrics_path),
        "aggregate_metrics_path": str(agg_metrics_path),
        "predictions_path": str(predictions_path),
    }

    return fold_metrics_df, aggregate_metrics_df, predictions_df, metadata


def generate_walk_forward_plots(
    fold_metrics_df: pd.DataFrame,
    agg_metrics_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    targets_cfg: Dict[str, Any],
    fig_dir: Path,
):
    """Generate diagnostic visual charts for walk-forward results."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)

    # 1. Fold-by-Fold MAE Comparison per Vessel Class
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for i, tgt in enumerate(targets_cfg.keys()):
        sub_df = fold_metrics_df[fold_metrics_df["target"] == tgt]
        sns.barplot(data=sub_df, x="fold", y="mae", hue="model", ax=axes[i], palette="tab10")
        axes[i].set_title(f"{tgt.upper()} - MAE by Walk-Forward Fold", fontweight="bold")
        axes[i].set_ylabel("MAE (Points)")
        axes[i].set_xlabel("Evaluation Fold (Chronological)")
        if i == 0:
            axes[i].legend(loc="upper left", title="Model")
        else:
            axes[i].legend().remove()

    plt.tight_layout()
    fig.savefig(fig_dir / "01_fold_by_fold_mae_comparison.png", dpi=200)
    plt.close(fig)

    # 2. Aggregate Mean MAE vs Mean Directional Accuracy
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.flatten()
    for i, tgt in enumerate(targets_cfg.keys()):
        sub_agg = agg_metrics_df[agg_metrics_df["target"] == tgt].sort_values(by="mean_mae")
        sns.barplot(data=sub_agg, x="mean_mae", y="model", ax=axes[i], palette="viridis")
        axes[i].set_title(f"{tgt.upper()} - Mean Walk-Forward MAE (Across 5 Folds)", fontweight="bold")
        axes[i].set_xlabel("Mean MAE (Lower is Better)")
        axes[i].set_ylabel("Model")

    plt.tight_layout()
    fig.savefig(fig_dir / "02_aggregate_model_ranking.png", dpi=200)
    plt.close(fig)

    # 3. Regime Out-of-Sample Predictions Across All 5 Folds (Continuous Time Series)
    dates = pd.to_datetime(pred_df[DATE_COLUMN])
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    for i, tgt in enumerate(targets_cfg.keys()):
        actual = pred_df[f"actual_{tgt}"]
        ridge = pred_df.get(f"pred_{tgt}_ridge", actual)
        xgb_p = pred_df.get(f"pred_{tgt}_xgboost", actual)
        persist = pred_df[f"pred_{tgt}_persistence"]

        axes[i].plot(dates, actual, label="Actual Level", color="black", lw=1.5)
        axes[i].plot(dates, ridge, label="Ridge (alpha=1.0)", color="#1f77b4", linestyle="--", alpha=0.8)
        axes[i].plot(dates, xgb_p, label="XGBoost", color="#2ca02c", linestyle=":", alpha=0.8)
        axes[i].plot(dates, persist, label="Persistence", color="gray", linestyle="-.", alpha=0.6)
        axes[i].set_title(f"{tgt.upper()} Out-of-Sample Walk-Forward Predictions (2015-08 to 2019-07)", fontweight="bold")
        axes[i].set_ylabel("Index Level")
        axes[i].legend(loc="upper left")

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    fig.savefig(fig_dir / "03_regime_error_across_time.png", dpi=200)
    plt.close(fig)
