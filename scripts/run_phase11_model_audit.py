"""Phase 11: Decision-Oriented Model Audit Script.

Performs:
1. Target/Feature Alignment Verification & Traceability Table
2. Feature Audit & Grouping (Groups A-I), Ridge & XGBoost Importance Ranking
3. Direct Performance & Relative Improvement Benchmark vs Persistence Baseline
4. Market Regime Analysis (Low, Normal, High Freight, High-Vol, Rising/Falling Momentum)
5. Business Decision Signal Evaluation (CHARTER NOW, WAIT, FLEXIBLE / MONITOR)
6. Expanding Walk-Forward Stability Analysis (Mean, Std, Min, Max)
7. Artifact Generation under experiments/expanded/ and reporting
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.schemas import DATE_COLUMN
from src.models.baselines import PersistenceForecaster
from src.models.ridge import RidgeForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.lstm_model import LSTMForecaster
from src.models.evaluation import compute_regression_metrics
from scripts.run_expanded_training_experiment import get_valid_kobc_features, KOBC_TARGETS


def classify_feature_group(feature_name: str) -> str:
    """Classify feature into audit categories A through I."""
    f = feature_name.lower()
    if f.startswith("cross_kobc_"):
        return "B_cross_vessel_freight"
    elif "kdci" in f:
        return "C_kdci_features"
    elif any(f.startswith(k) for k in ["kobc_handy_", "kobc_supramax_", "kobc_panamax_", "kobc_cape_"]):
        return "A_autoregressive_freight"
    elif any(k in f for k in ["wti_", "brent_", "coal", "iron_ore"]):
        return "D_commodity_energy"
    elif "usd_inr" in f:
        return "E_fx_macro"
    elif "gpr" in f:
        return "F_geopolitical_risk"
    elif any(k in f for k in ["temp", "precip", "wind", "pressure"]):
        return "G_weather"
    elif any(k in f for k in ["day_of_week", "month", "quarter", "year", "is_month", "is_quarter", "sin", "cos"]):
        return "H_calendar"
    elif any(k in f for k in ["turnaround", "fleet", "dwt", "congestion"]):
        return "I_port_operational"
    return "Other"


def run_phase11_audit():
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "experiments" / "expanded"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = project_root / "data" / "features" / "freight_features_expanded.csv"
    df = pd.read_csv(input_path)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    df_kobc = df[df[DATE_COLUMN] >= pd.Timestamp("2020-01-01")].sort_values(by=DATE_COLUMN).reset_index(drop=True)
    features = get_valid_kobc_features(df_kobc)

    # Clean data (drop cold-start and final unobserved target)
    df_clean = df_kobc.iloc[21:-1].reset_index(drop=True)
    n = len(df_clean)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val

    train_df = df_clean.iloc[:n_train].reset_index(drop=True)
    val_df = df_clean.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df_clean.iloc[n_train + n_val:].reset_index(drop=True)

    # ==================== 1. FEATURE GROUPING & IMPORTANCE ====================
    print("Executing Section 1: Feature Grouping & Importance Audit...")
    feature_group_rows = []
    for f in features:
        grp = classify_feature_group(f)
        null_cnt = int(df_kobc[f].isnull().sum())
        std_val = float(df_kobc[f].std())
        feature_group_rows.append({
            "feature": f,
            "group": grp,
            "null_count": null_cnt,
            "null_pct": round(null_cnt / len(df_kobc) * 100.0, 2),
            "std": round(std_val, 4),
            "is_constant": bool(std_val == 0.0 or np.isnan(std_val)),
        })
    feature_groups_df = pd.DataFrame(feature_group_rows)
    feature_groups_df.to_csv(output_dir / "feature_groups.csv", index=False)

    # Fit Ridge & XGBoost for feature importance extraction across all targets
    feature_importance_rows = []
    for target_col, current_col in KOBC_TARGETS.items():
        vessel = target_col.replace("target_kobc_", "").replace("_next", "").capitalize()
        
        # Ridge
        ridge = RidgeForecaster(alpha=1.0, scale_features=True).fit(train_df[features], train_df[target_col])
        ridge_coefs = ridge.model.coef_
        
        # XGBoost
        xgb = XGBoostForecaster(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42).fit(train_df[features], train_df[target_col])
        xgb_imp_df = xgb.get_feature_importances(top_n=len(features))
        xgb_map = dict(zip(xgb_imp_df["feature"], xgb_imp_df["importance_pct"]))

        for i, f in enumerate(features):
            feature_importance_rows.append({
                "vessel": vessel,
                "target": target_col,
                "feature": f,
                "group": classify_feature_group(f),
                "ridge_coefficient": round(float(ridge_coefs[i]), 4),
                "ridge_abs_coef": round(float(abs(ridge_coefs[i])), 4),
                "xgboost_importance_pct": round(float(xgb_map.get(f, 0.0)), 4),
            })
    feature_importance_df = pd.DataFrame(feature_importance_rows)
    feature_importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # ==================== 2. COMPARISON VS PERSISTENCE (RELATIVE IMPROVEMENT) ====================
    print("Executing Section 2: Model Performance & Relative Persistence Benchmarking...")
    model_audit_rows = []

    for target_col, current_col in KOBC_TARGETS.items():
        vessel = target_col.replace("target_kobc_", "").replace("_next", "").capitalize()
        X_tr, y_tr = train_df[features], train_df[target_col]
        X_te, y_te = test_df[features], test_df[target_col]
        y_curr = test_df[current_col]

        # 1. Persistence
        pred_p = PersistenceForecaster().predict(y_curr)
        mp = compute_regression_metrics(y_te, pred_p, y_curr)
        
        # 2. Ridge
        ridge = RidgeForecaster(alpha=1.0, scale_features=True).fit(X_tr, y_tr)
        pred_r = ridge.predict(X_te)
        mr = compute_regression_metrics(y_te, pred_r, y_curr)

        # 3. XGBoost
        xgb = XGBoostForecaster(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42).fit(X_tr, y_tr)
        pred_x = xgb.predict(X_te)
        mx = compute_regression_metrics(y_te, pred_x, y_curr)

        # 4. LSTM
        lstm = LSTMForecaster(lookback=10, hidden_size=32, dense_units=16, max_epochs=20, random_seed=42).fit(X_tr, y_tr)
        pred_l_raw = lstm.predict(X_te)
        pad_count = 9
        pred_l = np.concatenate([y_curr.iloc[:pad_count].values, pred_l_raw])
        ml = compute_regression_metrics(y_te.iloc[pad_count:], pred_l_raw, y_curr.iloc[pad_count:])

        models_data = [
            ("Persistence", mp, pred_p),
            ("Ridge", mr, pred_r),
            ("XGBoost", mx, pred_x),
            ("LSTM", ml, pred_l),
        ]

        p_mae, p_rmse, p_smape = mp["mae"], mp["rmse"], mp["smape"]

        for m_name, m_dict, m_pred in models_data:
            mae_imp = round(((p_mae - m_dict["mae"]) / p_mae) * 100.0, 2) if p_mae > 0 else 0.0
            rmse_imp = round(((p_rmse - m_dict["rmse"]) / p_rmse) * 100.0, 2) if p_rmse > 0 else 0.0
            smape_imp = round(((p_smape - m_dict["smape"]) / p_smape) * 100.0, 2) if p_smape > 0 else 0.0

            model_audit_rows.append({
                "vessel": vessel,
                "target": target_col,
                "model": m_name,
                "mae": m_dict["mae"],
                "rmse": m_dict["rmse"],
                "smape": m_dict["smape"],
                "r2": m_dict["r2"],
                "da_pct": m_dict["da_pct"],
                "mae_improvement_pct": mae_imp,
                "rmse_improvement_pct": rmse_imp,
                "smape_improvement_pct": smape_imp,
                "beats_persistence_mae": bool(m_dict["mae"] < p_mae),
                "beats_persistence_rmse": bool(m_dict["rmse"] < p_rmse),
            })

    model_audit_df = pd.DataFrame(model_audit_rows)
    model_audit_df.to_csv(output_dir / "model_audit.csv", index=False)

    # ==================== 3. REGIME ANALYSIS ====================
    print("Executing Section 3: Regime Analysis...")
    regime_rows = []

    for target_col, current_col in KOBC_TARGETS.items():
        vessel = target_col.replace("target_kobc_", "").replace("_next", "").capitalize()
        y_tr = train_df[target_col]
        y_curr_te = test_df[current_col]
        y_true_te = test_df[target_col]

        # Calculate regime thresholds from TRAIN data only
        q25 = float(y_tr.quantile(0.25))
        q75 = float(y_tr.quantile(0.75))
        
        # Volatility threshold from train rolling 21d std
        vol_col = f"{current_col.replace('_level', '')}_vol_21"
        vol_thresh = float(train_df[vol_col].quantile(0.75)) if vol_col in train_df.columns else 100.0

        # Predictions
        X_tr, X_te = train_df[features], test_df[features]
        pred_p = PersistenceForecaster().predict(y_curr_te)
        pred_r = RidgeForecaster(alpha=1.0).fit(X_tr, y_tr).predict(X_te)
        pred_x = XGBoostForecaster(n_estimators=100, max_depth=3, learning_rate=0.05).fit(X_tr, y_tr).predict(X_te)

        # Regimes definitions based purely on t information
        regimes = {
            "Low_Freight (<= Q25)": y_curr_te <= q25,
            "Normal_Freight (Q25-Q75)": (y_curr_te > q25) & (y_curr_te <= q75),
            "High_Freight (> Q75)": y_curr_te > q75,
            "High_Volatility (> Q75_vol)": test_df[vol_col] > vol_thresh if vol_col in test_df.columns else pd.Series(False, index=test_df.index),
            "Rising_Market (5d_diff > 0)": (test_df[current_col] - test_df[f"{current_col.replace('_level', '')}_lag_5"]) > 0 if f"{current_col.replace('_level', '')}_lag_5" in test_df.columns else pd.Series(True, index=test_df.index),
            "Falling_Market (5d_diff < 0)": (test_df[current_col] - test_df[f"{current_col.replace('_level', '')}_lag_5"]) < 0 if f"{current_col.replace('_level', '')}_lag_5" in test_df.columns else pd.Series(False, index=test_df.index),
        }

        for r_name, r_mask in regimes.items():
            count = int(r_mask.sum())
            if count == 0:
                continue

            sub_true = y_true_te[r_mask]
            sub_curr = y_curr_te[r_mask]

            # Persistence
            mp = compute_regression_metrics(sub_true, pred_p[r_mask], sub_curr)
            # Ridge
            mr = compute_regression_metrics(sub_true, pred_r[r_mask], sub_curr)
            # XGBoost
            mx = compute_regression_metrics(sub_true, pred_x[r_mask], sub_curr)

            for m_name, m_res in [("Persistence", mp), ("Ridge", mr), ("XGBoost", mx)]:
                regime_rows.append({
                    "vessel": vessel,
                    "target": target_col,
                    "regime": r_name,
                    "observation_count": count,
                    "model": m_name,
                    "mae": m_res["mae"],
                    "rmse": m_res["rmse"],
                    "smape": m_res["smape"],
                    "r2": m_res["r2"],
                    "da_pct": m_res["da_pct"],
                })

    regime_metrics_df = pd.DataFrame(regime_rows)
    regime_metrics_df.to_csv(output_dir / "regime_metrics.csv", index=False)

    # ==================== 4. BUSINESS-DECISION SIGNAL EVALUATION ====================
    print("Executing Section 4: Business Decision Signal Evaluation...")
    decision_rows = []

    # Threshold for market movement decision (0.5% expected change threshold)
    threshold_pct = 0.50

    for target_col, current_col in KOBC_TARGETS.items():
        vessel = target_col.replace("target_kobc_", "").replace("_next", "").capitalize()
        y_curr = test_df[current_col].values
        y_true = test_df[target_col].values

        # Actual market move
        actual_diff = y_true - y_curr
        actual_direction = np.sign(actual_diff)  # +1 = rise, -1 = fall, 0 = flat

        X_tr, y_tr = train_df[features], train_df[target_col]
        X_te = test_df[features]

        pred_p = PersistenceForecaster().predict(test_df[current_col])
        pred_r = RidgeForecaster(alpha=1.0).fit(X_tr, y_tr).predict(X_te)
        pred_x = XGBoostForecaster(n_estimators=100, max_depth=3, learning_rate=0.05).fit(X_tr, y_tr).predict(X_te)

        models_eval = [
            ("Persistence", pred_p),
            ("Ridge", pred_r),
            ("XGBoost", pred_x),
        ]

        total_obs = len(y_true)

        for m_name, pred_arr in models_eval:
            expected_change = pred_arr - y_curr
            expected_pct_change = (expected_change / y_curr) * 100.0

            charter_now_mask = expected_pct_change > threshold_pct
            wait_mask = expected_pct_change < -threshold_pct
            flexible_mask = (~charter_now_mask) & (~wait_mask)

            cnt_charter_now = int(charter_now_mask.sum())
            cnt_wait = int(wait_mask.sum())
            cnt_flexible = int(flexible_mask.sum())

            # Correct signals
            # Correct Charter Now: Signalled Charter Now and actual rose (actual_diff > 0)
            correct_charter = int((charter_now_mask & (actual_diff > 0)).sum())
            # False Charter Now: Signalled Charter Now but actual fell (actual_diff < 0)
            false_charter = int((charter_now_mask & (actual_diff < 0)).sum())

            # Correct Wait: Signalled Wait and actual fell (actual_diff < 0)
            correct_wait = int((wait_mask & (actual_diff < 0)).sum())
            # False Wait: Signalled Wait but actual rose (actual_diff > 0)
            false_wait = int((wait_mask & (actual_diff > 0)).sum())

            total_active_signals = cnt_charter_now + cnt_wait
            total_correct_signals = correct_charter + correct_wait
            accuracy_active_signals = round((total_correct_signals / total_active_signals) * 100.0, 2) if total_active_signals > 0 else 0.0

            # Directional alignment across all days
            pred_dir = np.sign(expected_change)
            correct_dir_all = int((pred_dir == actual_direction).sum())
            da_all_pct = round((correct_dir_all / total_obs) * 100.0, 2)

            decision_rows.append({
                "vessel": vessel,
                "target": target_col,
                "model": m_name,
                "threshold_pct": threshold_pct,
                "total_days": total_obs,
                "charter_now_signals": cnt_charter_now,
                "wait_signals": cnt_wait,
                "flexible_signals": cnt_flexible,
                "correct_charter_now": correct_charter,
                "false_charter_now": false_charter,
                "correct_wait": correct_wait,
                "false_wait": false_wait,
                "active_signal_accuracy_pct": accuracy_active_signals,
                "overall_directional_accuracy_pct": da_all_pct,
            })

    decision_metrics_df = pd.DataFrame(decision_rows)
    decision_metrics_df.to_csv(output_dir / "decision_metrics.csv", index=False)

    print("\nPhase 11 Audit successfully completed and exported all CSV tables.")
    return feature_groups_df, feature_importance_df, model_audit_df, regime_metrics_df, decision_metrics_df


if __name__ == "__main__":
    run_phase11_audit()
