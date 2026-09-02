"""XGBoost regression forecasting model pipeline."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import xgboost as xgb


class XGBoostForecaster:
    """XGBoost regression forecaster for dry-bulk freight time-series."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 3.0,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "objective": "reg:squarederror",
            "eval_metric": "mae",
        }
        self.name = f"XGBoost_d{max_depth}_lr{learning_rate}_n{n_estimators}"
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
        verbose: bool = False,
    ) -> "XGBoostForecaster":
        """Train XGBoost model strictly on training observations.

        Args:
            X_train: Training feature DataFrame.
            y_train: Target series for training.
            eval_set: Optional validation pair list for early evaluation.
            verbose: Verbosity flag during boosting iterations.

        Returns:
            XGBoostForecaster: Fitted model instance.
        """
        self.feature_names = list(X_train.columns)
        self.model = xgb.XGBRegressor(**self.params)

        # Convert eval_set to numpy arrays if provided
        eval_pairs = None
        if eval_set is not None:
            eval_pairs = [(X.values, y.values) for X, y in eval_set]

        self.model.fit(
            X_train.values,
            y_train.values,
            eval_set=eval_pairs,
            verbose=verbose,
        )
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate forecasts using trained XGBoost trees.

        Args:
            X: Feature matrix.

        Returns:
            np.ndarray: Predicted freight rate array.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("XGBoostForecaster must be fitted before predict() is called.")

        X_input = X[self.feature_names].values
        return self.model.predict(X_input)

    def get_feature_importances(
        self, importance_type: str = "gain", top_n: int = 20
    ) -> pd.DataFrame:
        """Extract ranked feature importance scores from the booster.

        Args:
            importance_type: 'gain', 'weight', 'cover', or 'total_gain'.
            top_n: Number of top features to return.

        Returns:
            pd.DataFrame: Table with feature names and importance metrics.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted first.")

        # Extract native booster feature scores
        booster = self.model.get_booster()
        score_dict = booster.get_score(importance_type=importance_type)

        # Map f0, f1... or feature names
        importances = []
        for i, fname in enumerate(self.feature_names):
            f_key = f"f{i}"
            # booster might use f_key or actual name depending on feature name passing
            val = score_dict.get(fname, score_dict.get(f_key, 0.0))
            importances.append({"feature": fname, "importance": float(val)})

        imp_df = pd.DataFrame(importances)
        total_imp = imp_df["importance"].sum()
        if total_imp > 0:
            imp_df["importance_pct"] = (imp_df["importance"] / total_imp) * 100.0
        else:
            imp_df["importance_pct"] = 0.0

        imp_df = imp_df.sort_values(by="importance", ascending=False).reset_index(drop=True)
        return imp_df.head(top_n)

    def save_model(self, file_path: Union[str, Path]) -> None:
        """Save model artifact in native JSON format.

        Args:
            file_path: Destination path.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load_model(self, file_path: Union[str, Path], feature_names: List[str]) -> "XGBoostForecaster":
        """Load saved XGBoost artifact.

        Args:
            file_path: Path to model JSON artifact.
            feature_names: List of expected input feature names.

        Returns:
            XGBoostForecaster: Loaded model instance.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {path.resolve()}")
        self.feature_names = list(feature_names)
        self.model = xgb.XGBRegressor(**self.params)
        self.model.load_model(str(path))
        self.is_fitted = True
        return self
