"""Simple heuristic forecasting baselines for freight time series."""

from typing import Union, List
import pandas as pd
import numpy as np


class PersistenceForecaster:
    """Naive Persistence Baseline: Predicts next observed trading-day value as the current level.
    
    Mathematical formula:
        y_hat_{t+1} = y_t
    """

    def __init__(self):
        self.name = "Persistence"

    def predict(self, current_level: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Generate 1-step-ahead persistence forecast.

        Args:
            current_level: Series or array of current freight levels y_t.

        Returns:
            np.ndarray: Forecast array y_hat_{t+1}.
        """
        if isinstance(current_level, pd.Series):
            return current_level.values.astype(float)
        return np.asarray(current_level, dtype=float)


class MovingAverageForecaster:
    """Simple Moving Average Baseline: Predicts next value as the mean of recent W trading days.

    Mathematical formula:
        y_hat_{t+1} = (1 / W) * sum_{i=0}^{W-1} y_{t-i}
    """

    def __init__(self, window: int = 5):
        if window < 1:
            raise ValueError(f"Moving average window must be >= 1, got {window}")
        self.window = window
        self.name = f"MA_{window}"

    def predict(self, historical_series: pd.Series) -> pd.Series:
        """Generate causal moving average forecasts using rolling mean.

        Args:
            historical_series: Time series of past and current observations up to t.

        Returns:
            pd.Series: Forecast series y_hat_{t+1}.
        """
        # Strictly backward-looking (center=False, uses only observations up to t)
        return historical_series.rolling(window=self.window, min_periods=1, center=False).mean()
