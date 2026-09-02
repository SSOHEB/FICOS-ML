"""Forecast uncertainty quantification and prediction interval generation."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd


class ResidualUncertaintyEstimator:
    """Estimates empirical prediction intervals from historical out-of-sample forecasting errors."""

    def __init__(self, residuals_by_target: Optional[Dict[str, np.ndarray]] = None):
        self.residuals_by_target: Dict[str, np.ndarray] = residuals_by_target or {}
        self.quantiles_by_target: Dict[str, Dict[str, float]] = {}
        if residuals_by_target:
            self._compute_quantiles()

    @classmethod
    def from_walk_forward_predictions(
        cls, predictions_path: Union[str, Path] = "experiments/phase8/predictions.csv"
    ) -> "ResidualUncertaintyEstimator":
        """Construct uncertainty estimator using historical walk-forward out-of-sample residuals.

        Args:
            predictions_path: Path to Phase 8 predictions.csv.

        Returns:
            ResidualUncertaintyEstimator: Fitted estimator instance.
        """
        path = Path(predictions_path)
        if not path.exists():
            raise FileNotFoundError(f"Walk-forward predictions not found at: {path.resolve()}")

        df = pd.read_csv(path)
        residuals_dict = {}

        targets = ["bdi_hsi", "bdi_si", "bdi_pi", "bdi_ci"]
        for tgt in targets:
            actual_col = f"actual_{tgt}"
            pred_col = f"pred_{tgt}_ridge"
            if actual_col in df.columns and pred_col in df.columns:
                # Error: e_t = y_{t+1} - \hat{y}_{t+1}
                res = df[actual_col].values - df[pred_col].values
                valid_res = res[~np.isnan(res)]
                residuals_dict[tgt] = valid_res

        estimator = cls(residuals_by_target=residuals_dict)
        return estimator

    def _compute_quantiles(self):
        """Compute empirical error quantiles (P10, P50, P90) per target index."""
        for tgt, res in self.residuals_by_target.items():
            if len(res) == 0:
                continue
            q10 = float(np.percentile(res, 10))
            q50 = float(np.percentile(res, 50))
            q90 = float(np.percentile(res, 90))
            mae = float(np.mean(np.abs(res)))
            std = float(np.std(res))
            self.quantiles_by_target[tgt] = {
                "q10": round(q10, 2),
                "q50": round(q50, 2),
                "q90": round(q90, 2),
                "mae": round(mae, 2),
                "std": round(std, 2),
            }

    def construct_prediction_interval(
        self, target_key: str, point_forecast: float, horizon_step: int = 1
    ) -> Dict[str, float]:
        """Construct empirical prediction interval for a given point forecast.

        Args:
            target_key: 'bdi_hsi', 'bdi_si', 'bdi_pi', or 'bdi_ci'.
            point_forecast: Expected point prediction level.
            horizon_step: Forecast step ahead h >= 1.

        Returns:
            Dict[str, float]: 'lower_p10', 'point_p50', 'upper_p90', 'uncertainty_range'.
        """
        # Fallback default heuristics if no residuals loaded
        defaults = {
            "bdi_hsi": {"q10": -3.5, "q50": 0.0, "q90": 3.5, "std": 2.7},
            "bdi_si": {"q10": -7.5, "q50": 0.0, "q90": 7.5, "std": 5.9},
            "bdi_pi": {"q10": -18.0, "q50": 0.0, "q90": 18.0, "std": 13.5},
            "bdi_ci": {"q10": -110.0, "q50": 0.0, "q90": 110.0, "std": 86.0},
        }

        q_info = self.quantiles_by_target.get(target_key, defaults.get(target_key, {"q10": -10.0, "q50": 0.0, "q90": 10.0, "std": 8.0}))
        
        # Scale uncertainty by sqrt(h) for multi-step horizon expansion
        scale_h = np.sqrt(max(1, horizon_step))

        lower = point_forecast + q_info["q10"] * scale_h
        upper = point_forecast + q_info["q90"] * scale_h

        # Prevent negative index levels
        lower = max(0.0, lower)
        upper = max(0.0, upper)

        return {
            "lower_p10": round(lower, 2),
            "point_p50": round(point_forecast, 2),
            "upper_p90": round(upper, 2),
            "interval_width": round(upper - lower, 2),
        }
