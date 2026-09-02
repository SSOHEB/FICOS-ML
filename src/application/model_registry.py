"""Production model loader and feature schema validator for FICOS."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import yaml
import numpy as np
import pandas as pd

from src.data.schemas import DATE_COLUMN
from src.models.ridge import RidgeForecaster


class ModelRegistry:
    """Manages loading, feature verification, and inference for production forecasting models."""

    def __init__(self, config_path: Union[str, Path] = "configs/models.yaml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found at: {self.config_path.resolve()}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.features_path = Path(self.config.get("data", {}).get("features_path", "data/features/freight_features.csv"))
        self.targets_cfg = self.config.get("targets", {})
        self.expected_feature_cols: Optional[List[str]] = None
        self.models: Dict[str, RidgeForecaster] = {}
        self.is_initialized = False

    def initialize(self) -> None:
        """Fit or load production Ridge models and register expected feature schemas."""
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found at: {self.features_path.resolve()}")

        df_feat = pd.read_csv(self.features_path)
        df_feat[DATE_COLUMN] = pd.to_datetime(df_feat[DATE_COLUMN])
        df_feat = df_feat.sort_values(by=DATE_COLUMN).reset_index(drop=True)

        target_col_names = [info["target_col"] for info in self.targets_cfg.values()]
        exclude_cols = [DATE_COLUMN, "is_bdi_trading_day"] + target_col_names
        self.expected_feature_cols = [c for c in df_feat.columns if c not in exclude_cols]

        # Fit production Ridge models on valid training observations
        # Using cold-start drop
        cold_start = int(self.config.get("data", {}).get("drop_initial_cold_start", 21))
        train_df = df_feat.iloc[cold_start:].copy()
        train_df = train_df[train_df[target_col_names[0]].notnull()].reset_index(drop=True)

        X_train = train_df[self.expected_feature_cols]

        for target_key, target_info in self.targets_cfg.items():
            target_col = target_info["target_col"]
            y_train = train_df[target_col]

            model = RidgeForecaster(alpha=1.0, scale_features=True)
            model.fit(X_train, y_train)
            self.models[target_key] = model

        self.is_initialized = True

    def validate_feature_schema(self, X: pd.DataFrame) -> None:
        """Verify that input feature DataFrame matches the expected production schema."""
        if not self.is_initialized:
            raise RuntimeError("ModelRegistry must be initialized before validating feature schema.")

        missing_cols = [c for c in self.expected_feature_cols if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Feature schema mismatch. Missing {len(missing_cols)} required columns: {missing_cols[:5]}...")

    def predict_one_step(self, target_key: str, X: pd.DataFrame) -> float:
        """Generate 1-step ahead forecast for target index."""
        if not self.is_initialized:
            self.initialize()

        if target_key not in self.models:
            raise ValueError(f"Target '{target_key}' not found in registry. Available: {list(self.models.keys())}")

        self.validate_feature_schema(X)
        X_aligned = X[self.expected_feature_cols]
        preds = self.models[target_key].predict(X_aligned)
        return float(preds[-1])
