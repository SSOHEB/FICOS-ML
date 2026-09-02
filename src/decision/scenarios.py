"""Historical scenario replay and evaluation for the charter decision engine."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.schemas import DATE_COLUMN
from src.decision.uncertainty import ResidualUncertaintyEstimator
from src.decision.charter import CharterDecisionEngine, CharterDecisionRequest, CharterRecommendation


def run_phase9_historical_scenarios(
    config_path: Union[str, Path] = "configs/models.yaml",
    ports_config_path: Union[str, Path] = "configs/ports.yaml",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Execute historical replay scenario analysis using strictly contemporaneous information.

    Args:
        config_path: Path to models.yaml.
        ports_config_path: Path to ports.yaml.

    Returns:
        Tuple[scenarios_df, recommendations_df, cost_comparison_df, metadata].
    """
    cfg_path = Path(config_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    feat_path = Path(cfg.get("data", {}).get("features_path", "data/features/freight_features.csv"))
    if not feat_path.exists():
        raise FileNotFoundError(f"Features file not found at: {feat_path.resolve()}")

    df_feat = pd.read_csv(feat_path)
    df_feat[DATE_COLUMN] = pd.to_datetime(df_feat[DATE_COLUMN])
    df_feat = df_feat.sort_values(by=DATE_COLUMN).reset_index(drop=True)

    # 1. Initialize Uncertainty Estimator and Decision Engine
    uncertainty_estimator = ResidualUncertaintyEstimator.from_walk_forward_predictions(
        "experiments/phase8/predictions.csv"
    )
    engine = CharterDecisionEngine(
        ports_config_path=ports_config_path,
        uncertainty_estimator=uncertainty_estimator,
        threshold_cost_saving_pct=2.0,
        threshold_price_rise_pct=1.5,
    )

    out_dir = Path(cfg.get("output", {}).get("phase9_experiment_dir", "experiments/phase9"))
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 2. Define 10 Historical Replay Scenarios
    scenario_definitions = [
        {
            "scenario_id": "SCN_01_PARADIP_COAL_2016_TROUGH",
            "date": "2016-02-15",
            "cargo_type": "Coking Coal",
            "cargo_quantity_mt": 75000.0,
            "origin": "Gladstone, Australia",
            "destination_port": "paradip",
            "voyage_duration_days": 18.0,
            "laycan_days_allowed": 7,
            "context": "Market Crash / Record Low Freight Levels",
        },
        {
            "scenario_id": "SCN_02_VIZAG_COAL_2016_RECOVERY",
            "date": "2016-08-10",
            "cargo_type": "Thermal Coal",
            "cargo_quantity_mt": 55000.0,
            "origin": "Richards Bay, South Africa",
            "destination_port": "vizag",
            "voyage_duration_days": 14.0,
            "laycan_days_allowed": 7,
            "context": "Early Freight Recovery Momentum",
        },
        {
            "scenario_id": "SCN_03_GANGAVARAM_ORE_2017_BULL",
            "date": "2017-06-20",
            "cargo_type": "Iron Ore",
            "cargo_quantity_mt": 170000.0,
            "origin": "Tubarao, Brazil",
            "destination_port": "gangavaram",
            "voyage_duration_days": 28.0,
            "laycan_days_allowed": 7,
            "context": "Capesize Cyclical Bull Rally",
        },
        {
            "scenario_id": "SCN_04_DHAMRA_COAL_2017_EXPANSION",
            "date": "2017-10-16",
            "cargo_type": "Coking Coal",
            "cargo_quantity_mt": 75000.0,
            "origin": "Hay Point, Australia",
            "destination_port": "dhamra",
            "voyage_duration_days": 17.0,
            "laycan_days_allowed": 7,
            "context": "Sustained Freight Expansion Cycle",
        },
        {
            "scenario_id": "SCN_05_GOPALPUR_LIMESTONE_2018_PEAK",
            "date": "2018-04-12",
            "cargo_type": "Limestone",
            "cargo_quantity_mt": 45000.0,
            "origin": "Mina Saqr, UAE",
            "destination_port": "gopalpur",
            "voyage_duration_days": 10.0,
            "laycan_days_allowed": 5,
            "context": "Draft-Constrained Port / Supramax Allocation",
        },
        {
            "scenario_id": "SCN_06_HALDIA_COAL_2018_DRAFT_RESTRICTED",
            "date": "2018-07-18",
            "cargo_type": "Thermal Coal",
            "cargo_quantity_mt": 30000.0,
            "origin": "Tanjung Bara, Indonesia",
            "destination_port": "haldia",
            "voyage_duration_days": 8.0,
            "laycan_days_allowed": 5,
            "context": "Shallow Draft Port / Handysize Mandatory",
        },
        {
            "scenario_id": "SCN_07_PARADIP_COAL_2018_TARIFF_SHOCK",
            "date": "2018-09-25",
            "cargo_type": "Coking Coal",
            "cargo_quantity_mt": 75000.0,
            "origin": "Gladstone, Australia",
            "destination_port": "paradip",
            "voyage_duration_days": 18.0,
            "laycan_days_allowed": 7,
            "context": "High Volatility / US-China Tariff Dispute",
        },
        {
            "scenario_id": "SCN_08_VIZAG_ORE_2019_BRUMADINHO_SLUMP",
            "date": "2019-02-14",
            "cargo_type": "Iron Ore",
            "cargo_quantity_mt": 150000.0,
            "origin": "Port Hedland, Australia",
            "destination_port": "vizag",
            "voyage_duration_days": 15.0,
            "laycan_days_allowed": 7,
            "context": "Post-Dam Disaster Market Shock",
        },
        {
            "scenario_id": "SCN_09_DHAMRA_COAL_2019_PRE_MONSOON",
            "date": "2019-05-20",
            "cargo_type": "Coking Coal",
            "cargo_quantity_mt": 70000.0,
            "origin": "Newcastle, Australia",
            "destination_port": "dhamra",
            "voyage_duration_days": 18.0,
            "laycan_days_allowed": 7,
            "context": "Pre-Monsoon Restocking",
        },
        {
            "scenario_id": "SCN_10_GANGAVARAM_ORE_2019_CAPESIZE_SQUEEZE",
            "date": "2019-07-10",
            "cargo_type": "Iron Ore",
            "cargo_quantity_mt": 175000.0,
            "origin": "Tubarao, Brazil",
            "destination_port": "gangavaram",
            "voyage_duration_days": 28.0,
            "laycan_days_allowed": 7,
            "context": "Extreme July 2019 Capesize Freight Squeeze",
        },
    ]

    scenarios_list = []
    recommendations_list = []
    cost_comp_list = []

    for scn in scenario_definitions:
        target_date = pd.to_datetime(scn["date"])
        # Find closest available trading observation <= target_date
        past_df = df_feat[df_feat[DATE_COLUMN] <= target_date]
        if past_df.empty:
            continue
        row = past_df.iloc[-1]
        actual_decision_date = row[DATE_COLUMN].strftime("%Y-%m-%d")

        req = CharterDecisionRequest(
            cargo_type=scn["cargo_type"],
            cargo_quantity_mt=scn["cargo_quantity_mt"],
            origin=scn["origin"],
            destination_port=scn["destination_port"],
            decision_date=actual_decision_date,
            laycan_days_allowed=scn["laycan_days_allowed"],
            voyage_duration_days=scn["voyage_duration_days"],
        )

        # Preliminary vessel selection to identify correct target index
        feas_vessels, _ = engine.evaluate_vessel_feasibility(req.cargo_quantity_mt, req.destination_port)
        opt_vessel = engine.select_optimal_vessel(feas_vessels, req.cargo_quantity_mt)
        target_key = engine.vessels[opt_vessel]["target_key"]

        current_freight = float(row[f"{target_key}_level"])
        
        # Extract contemporaneous signals from features
        # 5-day difference momentum converted to daily drift pct
        mom_col = f"{target_key}_diff_5"
        diff_5 = float(row[mom_col]) if mom_col in row else 0.0
        drift_pct = (diff_5 / (5.0 * max(1.0, current_freight))) * 100.0

        vol_col = f"{target_key}_return_vol_7"
        rolling_vol = float(row[vol_col]) if vol_col in row and not np.isnan(row[vol_col]) else 0.02

        gpr_col = "gpr_spike_ratio_ma30"
        gpr_ratio = float(row[gpr_col]) if gpr_col in row and not np.isnan(row[gpr_col]) else 1.0

        precip_col = "precip_mm_lag_1"
        precip = float(row[precip_col]) if precip_col in row and not np.isnan(row[precip_col]) else 0.0
        weather_alert = precip > 25.0

        # Run Recommendation
        rec = engine.recommend_charter(
            request=req,
            current_freight_index=current_freight,
            expected_drift_pct_per_day=drift_pct,
            rolling_volatility=rolling_vol,
            gpr_spike_ratio=gpr_ratio,
            weather_alert=weather_alert,
        )

        # Record outputs
        scenarios_list.append({
            "scenario_id": scn["scenario_id"],
            "requested_date": scn["date"],
            "historical_decision_date": actual_decision_date,
            "cargo_type": scn["cargo_type"],
            "cargo_quantity_mt": scn["cargo_quantity_mt"],
            "origin": scn["origin"],
            "destination_port": scn["destination_port"],
            "context": scn["context"],
        })

        recommendations_list.append({
            "scenario_id": scn["scenario_id"],
            "decision_date": actual_decision_date,
            "charter_action": rec.charter_action,
            "recommended_vessel": rec.recommended_vessel,
            "current_freight_index": rec.current_freight_index,
            "optimal_entry_day": rec.optimal_entry_day,
            "expected_cost_now_usd": rec.expected_cost_now_usd,
            "expected_cost_optimal_usd": rec.expected_cost_optimal_usd,
            "estimated_savings_usd": rec.estimated_savings_usd,
            "estimated_savings_pct": rec.estimated_savings_pct,
            "risk_level": rec.risk_level,
            "primary_reason": rec.reasons[0] if rec.reasons else "",
            "risk_summary": "; ".join(rec.risk_reasons),
        })

        cost_comp_list.append({
            "scenario_id": scn["scenario_id"],
            "vessel": rec.recommended_vessel,
            "cargo_mt": req.cargo_quantity_mt,
            "charter_now_cost_usd": rec.expected_cost_now_usd,
            "optimal_cost_usd": rec.expected_cost_optimal_usd,
            "net_savings_usd": rec.estimated_savings_usd,
            "net_savings_pct": rec.estimated_savings_pct,
            "action": rec.charter_action,
        })

    scenarios_df = pd.DataFrame(scenarios_list)
    recommendations_df = pd.DataFrame(recommendations_list)
    cost_comparison_df = pd.DataFrame(cost_comp_list)

    # Save CSVs
    scenarios_path = out_dir / "scenarios.csv"
    recs_path = out_dir / "recommendations.csv"
    cost_path = out_dir / "cost_comparison.csv"
    cfg_saved_path = out_dir / "configuration.yaml"

    scenarios_df.to_csv(scenarios_path, index=False)
    recommendations_df.to_csv(recs_path, index=False)
    cost_comparison_df.to_csv(cost_path, index=False)
    with open(cfg_saved_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # Generate Visuals
    generate_scenario_plots(recommendations_df, cost_comparison_df, figures_dir)

    metadata = {
        "num_scenarios_evaluated": len(scenarios_df),
        "actions_breakdown": dict(recommendations_df["charter_action"].value_counts()),
        "risk_breakdown": dict(recommendations_df["risk_level"].value_counts()),
        "mean_savings_pct": round(float(cost_comparison_df["net_savings_pct"].mean()), 2),
        "total_simulated_opportunity_usd": round(float(cost_comparison_df["net_savings_usd"].sum()), 2),
    }

    return scenarios_df, recommendations_df, cost_comparison_df, metadata


def generate_scenario_plots(
    recs_df: pd.DataFrame, cost_df: pd.DataFrame, fig_dir: Path
):
    """Generate diagnostic visual charts for Phase 9 historical replay scenarios."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)

    # 1. Action Breakdown & Risk Level Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    action_counts = recs_df["charter_action"].value_counts()
    ax1.pie(
        action_counts.values,
        labels=action_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=["#1f77b4", "#2ca02c", "#ff7f0e"][:len(action_counts)],
    )
    ax1.set_title("Decision Action Distribution (10 Scenarios)", fontweight="bold")

    risk_counts = recs_df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["risk_level", "count"]
    sns.barplot(data=risk_counts, x="risk_level", y="count", hue="risk_level", ax=ax2, palette="Reds_r", legend=False)
    ax2.set_title("Risk Level Classification", fontweight="bold")
    ax2.set_ylabel("Scenario Count")

    plt.tight_layout()
    fig.savefig(fig_dir / "01_decision_actions_and_risk.png", dpi=200)
    plt.close(fig)

    # 2. Simulated Cost Comparison: Charter Now vs Optimal Expected
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(cost_df))
    width = 0.35

    ax.bar(x - width / 2, cost_df["charter_now_cost_usd"] / 1000, width, label="Charter Now Cost ($k)", color="#1f77b4")
    ax.bar(x + width / 2, cost_df["optimal_cost_usd"] / 1000, width, label="Optimal Timing Cost ($k)", color="#2ca02c")

    ax.set_ylabel("Voyage Cost ($ in Thousands)")
    ax.set_title("Historical Scenario Cost Comparison: Charter Now vs Optimal Timing", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i+1}\n({r['action']})" for i, r in cost_df.iterrows()], rotation=0)
    ax.legend()

    plt.tight_layout()
    fig.savefig(fig_dir / "02_charter_cost_comparison.png", dpi=200)
    plt.close(fig)
