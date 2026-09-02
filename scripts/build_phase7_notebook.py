"""Build the Phase 7 LSTM Jupyter Notebook."""

import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# FICOS (Freight Intelligence & Charter Optimization System)\n",
                "## Phase 7: LSTM Deep Learning Forecasting & Multi-Model Evaluation\n",
                "\n",
                "### 1. Objective\n",
                "This notebook implements, evaluates, and diagnoses **LSTM Deep Learning Forecasters** using a 21-day sliding lookback window for the 4 Baltic dry-bulk freight sub-indices (`HSI`, `SI`, `PI`, `CI`).\n",
                "\n",
                "**Core Research Question:**\n",
                "> Can a deep recurrent neural network improve upon the Phase 5 Ridge baseline and Phase 6 XGBoost model in 1-step-ahead freight rate forecasting?"
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
                "import torch\n",
                "\n",
                "# Add project root to sys.path\n",
                "repo_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                "if str(repo_root) not in sys.path:\n",
                "    sys.path.insert(0, str(repo_root))\n",
                "\n",
                "from src.models.evaluation import (\n",
                "    split_chronological_holdout,\n",
                "    compute_regression_metrics,\n",
                "    run_phase7_lstm_experiment,\n",
                ")\n",
                "from src.models.lstm_model import LSTMForecaster\n",
                "from src.data.schemas import DATE_COLUMN, TARGET_COLUMNS\n",
                "\n",
                "print(f'PyTorch Version: {torch.__version__}')\n",
                "sns.set_theme(style='whitegrid', font_scale=1.0)\n",
                "plt.rcParams['figure.figsize'] = (12, 6)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 3. Load Feature Matrix & Chronological 80/20 Holdout Split"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "feat_path = repo_root / 'data' / 'features' / 'freight_features.csv'\n",
                "df_feat = pd.read_csv(feat_path)\n",
                "df_feat[DATE_COLUMN] = pd.to_datetime(df_feat[DATE_COLUMN])\n",
                "\n",
                "train_df, test_df = split_chronological_holdout(df_feat, train_ratio=0.80, drop_initial_cold_start=21)\n",
                "print(f'Total Features Matrix: {df_feat.shape[0]:,} rows x {df_feat.shape[1]} columns')\n",
                "print(f'Train Set: {len(train_df):,} sessions ({train_df[\"date\"].min().date()} to {train_df[\"date\"].max().date()})')\n",
                "print(f'Test Set:  {len(test_df):,} sessions ({test_df[\"date\"].min().date()} to {test_df[\"date\"].max().date()})')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 4. Execute Full Phase 7 Benchmark Experiment"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "config_path = repo_root / 'configs' / 'models.yaml'\n",
                "metrics_df, pred_df, meta = run_phase7_lstm_experiment(config_path)\n",
                "print(f'Experiment Completed across {meta[\"features_count\"]} selected features and 21-day lookback.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 5. Multi-Model Benchmark Comparison Table (Persistence vs Ridge vs XGBoost vs LSTM)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "display(metrics_df) if 'display' in globals() else print(metrics_df.to_string(index=False))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 6. Pivot Summary: MAE and Directional Accuracy Across Models"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pivot_mae = metrics_df.pivot(index='model', columns='target', values='mae')\n",
                "print('MAE Across Models and Vessel Classes:')\n",
                "display(pivot_mae) if 'display' in globals() else print(pivot_mae)\n",
                "\n",
                "pivot_da = metrics_df.pivot(index='model', columns='target', values='da_pct')\n",
                "print('\\nDirectional Accuracy (%) Across Models and Vessel Classes:')\n",
                "display(pivot_da) if 'display' in globals() else print(pivot_da)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 7. Multi-Model Forecast Visualizations in Test Horizon"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "dates = pd.to_datetime(pred_df['date'])\n",
                "fig, axes = plt.subplots(2, 2, figsize=(16, 10))\n",
                "axes = axes.flatten()\n",
                "targets = ['bdi_hsi', 'bdi_si', 'bdi_pi', 'bdi_ci']\n",
                "\n",
                "for i, tgt in enumerate(targets):\n",
                "    axes[i].plot(dates, pred_df[f'actual_{tgt}'], label='Actual', color='black', lw=1.5)\n",
                "    if f'pred_{tgt}_ridge' in pred_df.columns:\n",
                "        axes[i].plot(dates, pred_df[f'pred_{tgt}_ridge'], label='Ridge', linestyle='--', color='#1f77b4', alpha=0.8)\n",
                "    if f'pred_{tgt}_xgboost' in pred_df.columns:\n",
                "        axes[i].plot(dates, pred_df[f'pred_{tgt}_xgboost'], label='XGBoost', linestyle=':', color='#2ca02c', alpha=0.8)\n",
                "    axes[i].plot(dates, pred_df[f'pred_{tgt}_lstm'], label='LSTM', color='#9467bd', lw=1.3)\n",
                "    axes[i].set_title(f'{tgt.upper()} Test Period Multi-Model Comparison', fontweight='bold')\n",
                "    axes[i].set_ylabel('Index Level')\n",
                "    axes[i].legend(loc='upper left')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 8. LSTM Residual Error Diagnostics"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(2, 2, figsize=(14, 8))\n",
                "axes = axes.flatten()\n",
                "for i, tgt in enumerate(targets):\n",
                "    actual = pred_df[f'actual_{tgt}']\n",
                "    lstm_p = pred_df[f'pred_{tgt}_lstm']\n",
                "    res = actual - lstm_p\n",
                "    sns.histplot(res, kde=True, ax=axes[i], color='#9467bd', bins=30)\n",
                "    axes[i].set_title(f'{tgt.upper()} LSTM Residual Distribution (std={res.std():.2f})', fontweight='bold')\n",
                "    axes[i].set_xlabel('Error (Points)')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 9. Empirical Findings & Next Steps\n",
                "1. **Ridge Remains Champion**: Linear regularization outperforms both tree-based models and recurrent neural networks on daily raw freight index levels.\n",
                "2. **Recurrent Smoothing Distortion**: LSTMs struggle with sample efficiency on ~1,350 daily sequences and introduce phase lag on unit-root persistence series.\n",
                "3. **Future Strategic Direction**: Walk-forward backtesting, differenced target modeling ($\\Delta Y_{t+1}$), and hybrid linear-tree ensembling."
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

out_path = Path("notebooks/phase7_lstm.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Generated {out_path} successfully.")
