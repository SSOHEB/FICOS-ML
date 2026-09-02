"""Ridge regression forecasting model pipeline."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class RidgeForecaster:
    """Causal Ridge Regression forecaster with train-only scaling and imputation."""

    def __init__(self, alpha: float = 1.0, scale_features: bool = True):
        self.alpha = alpha
        self.scale_features = scale_features
        self.name = f"Ridge_alpha_{alpha}"
        
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.model = Ridge(alpha=alpha, random_state=42)
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RidgeForecaster":
        """Fit preprocessing transformers and Ridge regression strictly on training data.

        Args:
            X_train: Training feature matrix.
            y_train: Training target series.

        Returns:
            RidgeForecaster: Fitted forecaster instance.
        """
        self.feature_names = list(X_train.columns)

        # 1. Fit imputer ONLY on training set
        self.imputer = SimpleImputer(strategy="median")
        X_imp = self.imputer.fit_transform(X_train)

        # 2. Fit scaler ONLY on training set
        if self.scale_features:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_imp)
        else:
            self.scaler = None
            X_scaled = X_imp

        # 3. Fit Ridge model on training set
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate forecasts using transformers fitted during training.

        Args:
            X: Feature matrix (test or validation).

        Returns:
            np.ndarray: Predicted target values.
        """
        if not self.is_fitted or self.imputer is None:
            raise RuntimeError("RidgeForecaster must be fitted before predict() is called.")

        # Transform using pre-fitted transformers (no re-fitting)
        X_imp = self.imputer.transform(X[self.feature_names])
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_imp)
        else:
            X_scaled = X_imp

        return self.model.predict(X_scaled)

    def get_feature_importances(self, top_n: int = 15) -> pd.DataFrame:
        """Return top positive and negative regression coefficients."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")
        
        coef_df = pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": self.model.coef_,
            "abs_coefficient": np.abs(self.model.coef_),
        }).sort_values(by="abs_coefficient", ascending=False).reset_index(drop=True)
        
        return coef_df.head(top_n)
