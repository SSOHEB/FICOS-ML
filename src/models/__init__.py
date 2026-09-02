"""Models and evaluation pipelines for FICOS ML."""

from src.models.baselines import PersistenceForecaster, MovingAverageForecaster
from src.models.ridge import RidgeForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.lstm_model import PyTorchLSTM, LSTMForecaster
from src.models.evaluation import (
    compute_regression_metrics,
    split_chronological_holdout,
    generate_walk_forward_folds,
    run_phase5_experiment,
    run_phase6_xgboost_experiment,
    run_phase7_lstm_experiment,
    run_phase8_walk_forward_experiment,
)

__all__ = [
    "PersistenceForecaster",
    "MovingAverageForecaster",
    "RidgeForecaster",
    "XGBoostForecaster",
    "PyTorchLSTM",
    "LSTMForecaster",
    "compute_regression_metrics",
    "split_chronological_holdout",
    "generate_walk_forward_folds",
    "run_phase5_experiment",
    "run_phase6_xgboost_experiment",
    "run_phase7_lstm_experiment",
    "run_phase8_walk_forward_experiment",
]
