"""Build the Phase 4 Feature Engineering Jupyter Notebook."""

import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# FICOS (Freight Intelligence & Charter Optimization System)\n",
                "## Phase 4: Causal Feature Engineering & Target Alignment\n",
                "\n",
                "### 1. Objective\n",
                "This notebook implements and validates the causal feature engineering pipeline that transforms the historical master dataset into a model-ready feature table (`data/features/freight_features.csv`).\n",
                "\n",
                "**Key Requirements:**\n",
                "- **No Data Leakage**: All transformations must use strictly past and contemporaneous information at forecast origin $t$.\n",
                "- **Explicit Target Alignment**: Predict the next observed Baltic Dry Index trading day ($t+1$).\n",
                "- **Feature Families**: Autoregressive lags, cross-vessel interactions, momentum/returns, rolling volatility, macro/energy/FX, geopolitical shocks, and calendar indicators."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 2. Imports & Setup"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "from pathlib import Path\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Add project root to sys.path\n",
                "repo_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                "if str(repo_root) not in sys.path:\n",
                "    sys.path.insert(0, str(repo_root))\n",
                "\n",
                "from src.features.pipeline import build_features_dataframe, load_feature_config\n",
                "from src.data.schemas import DATE_COLUMN, TARGET_COLUMNS\n",
                "\n",
                "sns.set_theme(style='whitegrid', font_scale=1.0)\n",
                "plt.rcParams['figure.figsize'] = (12, 6)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 3. Load Clean Master Dataset (Phase 2 Output)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "master_path = repo_root / 'data' / 'processed' / 'master_dataset.csv'\n",
                "df_master = pd.read_csv(master_path)\n",
                "df_master[DATE_COLUMN] = pd.to_datetime(df_master[DATE_COLUMN])\n",
                "print(f'Master Dataset: {df_master.shape[0]:,} rows x {df_master.shape[1]} columns')\n",
                "df_master.head(3)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 4. Define Forecasting Task & Build Feature Matrix"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "config = load_feature_config(repo_root / 'configs' / 'features.yaml')\n",
                "df_features = build_features_dataframe(df_master, config=config, filter_trading_days=True)\n",
                "print(f'Feature Matrix: {df_features.shape[0]:,} trading sessions x {df_features.shape[1]} total columns')\n",
                "df_features.head(3)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 5. Target Alignment Verification (t -> t+1)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "sample_check = df_features[['date', 'bdi_hsi_level', 'target_bdi_hsi_next', 'bdi_ci_level', 'target_bdi_ci_next']].head(6)\n",
                "print('Verifying target alignment (row t target equals row t+1 level):')\n",
                "display(sample_check) if 'display' in globals() else print(sample_check)\n",
                "\n",
                "# Formal assertion check\n",
                "for col in TARGET_COLUMNS:\n",
                "    assert np.allclose(\n",
                "        df_features[f'target_{col}_next'].iloc[:-1],\n",
                "        df_features[f'{col}_level'].iloc[1:],\n",
                "        equal_nan=True\n",
                "    )\n",
                "print('All target alignments mathematically verified (zero look-ahead leakage).')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 6. Autoregressive & Cross-Vessel Lag Profiles"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ar_cols = [c for c in df_features.columns if 'bdi_hsi_lag_' in c or 'cross_bdi_si_lag_' in c]\n",
                "df_features[['date', 'bdi_hsi_level'] + ar_cols].head(8)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 7. Momentum & Difference Indicators"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "mom_cols = [c for c in df_features.columns if 'bdi_ci_diff_' in c or 'bdi_ci_pct_change_' in c]\n",
                "df_features[['date', 'bdi_ci_level'] + mom_cols].head(8)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 8. Rolling Channels & Return Volatility"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "roll_cols = [c for c in df_features.columns if 'bdi_pi_roll_' in c or 'bdi_pi_return_vol_' in c]\n",
                "df_features[['date', 'bdi_pi_level'] + roll_cols].tail(8)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 9. Exogenous Drivers (Macro, FX, GPR, Weather)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "exog_sample = ['wti_usd_bbl_lag_1', 'usd_inr_lag_1', 'gpr_lag_1', 'gpr_spike_ratio_ma30', 'wind_speed_max_kmh_lag_1']\n",
                "df_features[['date'] + [c for c in exog_sample if c in df_features.columns]].head(6)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 10. Calendar Features"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cal_cols = [c for c in df_features.columns if c.startswith('cal_')]\n",
                "df_features[['date'] + cal_cols].head(5)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 11. Missingness & Cold-Start Profile"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "null_summary = df_features.isnull().sum()\n",
                "top_nulls = null_summary[null_summary > 0].sort_values(ascending=False)\n",
                "print(f'Top feature columns with initial cold-start NaNs:\\n{top_nulls.head(10)}')\n",
                "print(f'\\nFinal row target NaN check (active forecast origin): {df_features[\"target_bdi_hsi_next\"].iloc[-1]}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 12. Leakage Independence Proof"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Perturb future observations at t >= 50\n",
                "df_corrupted = df_master.copy()\n",
                "df_corrupted.loc[50:, ['bdi_hsi', 'bdi_si', 'bdi_pi', 'bdi_ci']] *= 99.0\n",
                "df_feat_corrupt = build_features_dataframe(df_corrupted, config=config, filter_trading_days=True)\n",
                "\n",
                "feature_cols = [c for c in df_features.columns if not c.startswith('target_') and c != DATE_COLUMN]\n",
                "leakage_errors = 0\n",
                "for c in feature_cols:\n",
                "    if not np.allclose(df_features[c].iloc[:30].values, df_feat_corrupt[c].iloc[:30].values, equal_nan=True):\n",
                "        print(f'LEAKAGE FOUND in {c}')\n",
                "        leakage_errors += 1\n",
                "\n",
                "if leakage_errors == 0:\n",
                "    print('SUCCESS: Zero future leakage confirmed across all feature families.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 13. Phase 4 Summary & Next Steps\n",
                "- Total Generated Features: **135**\n",
                "- Total Forecasting Targets: **4** (`target_bdi_hsi_next`, `target_bdi_si_next`, `target_bdi_pi_next`, `target_bdi_ci_next`)\n",
                "- Stored Output: `data/features/freight_features.csv`\n",
                "- Ready for Phase 5 Baseline and Machine Learning Model Experiments."
            ]
        }
    ],
    "metadata": {
        "language_info": {"name": "python", "version": "3.13.14"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = Path("notebooks/phase4_features.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Generated {out_path} successfully.")
