"""LSTM deep learning forecasting model pipeline for dry-bulk freight."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class PyTorchLSTM(nn.Module):
    """PyTorch LSTM architecture for single-step time-series forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        dense_units: int = 32,
        num_layers: int = 1,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc1 = nn.Linear(hidden_size, dense_units)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(dense_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, lookback, input_size)
        out, (hn, cn) = self.lstm(x)
        # Use last timestep representation: (batch_size, hidden_size)
        last_out = out[:, -1, :]
        drop_out = self.dropout(last_out)
        h = self.relu(self.fc1(drop_out))
        return self.fc_out(h).squeeze(-1)


class LSTMForecaster:
    """End-to-end LSTM Forecaster with train-only normalization, sequence builder, and early stopping."""

    def __init__(
        self,
        lookback: int = 21,
        hidden_size: int = 64,
        dense_units: int = 32,
        num_layers: int = 1,
        dropout: float = 0.15,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        max_epochs: int = 60,
        early_stopping_patience: int = 10,
        random_seed: int = 42,
    ):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.dense_units = dense_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = early_stopping_patience
        self.random_seed = random_seed

        self.name = f"LSTM_lb{lookback}_h{hidden_size}"
        self.model: Optional[PyTorchLSTM] = None
        self.imputer: Optional[SimpleImputer] = None
        self.scaler_x: Optional[StandardScaler] = None
        self.scaler_y: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def _set_seed(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

    @staticmethod
    def create_sequences(
        X: np.ndarray, y: Optional[np.ndarray], lookback: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Construct sliding 3D lookback sequences for LSTM input.

        Args:
            X: 2D array (T, num_features).
            y: Optional 1D array of targets (T,).
            lookback: Lookback window length W.

        Returns:
            Tuple[X_seq, y_seq]:
                X_seq: (N, lookback, num_features) where N = T - lookback + 1.
                y_seq: (N,) targets corresponding to the end of each lookback window.
        """
        T = len(X)
        if T < lookback:
            raise ValueError(f"Length of series ({T}) must be >= lookback ({lookback}).")

        X_seq = []
        y_seq = []
        for i in range(lookback - 1, T):
            X_seq.append(X[i - lookback + 1 : i + 1, :])
            if y is not None:
                y_seq.append(y[i])

        X_seq_arr = np.array(X_seq, dtype=np.float32)
        y_seq_arr = np.array(y_seq, dtype=np.float32) if y is not None else None
        return X_seq_arr, y_seq_arr

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        val_ratio: float = 0.15,
        verbose: bool = False,
    ) -> "LSTMForecaster":
        """Fit preprocessing transformers and train LSTM with chronological early stopping.

        Args:
            X_train: Training features DataFrame.
            y_train: Training target Series.
            val_ratio: Out-of-time chronological validation fraction from the training set.
            verbose: Print training progress per epoch.

        Returns:
            LSTMForecaster: Fitted model instance.
        """
        self._set_seed()
        self.feature_names = list(X_train.columns)

        # 1. Fit Imputer and Scaler ONLY on training features
        self.imputer = SimpleImputer(strategy="median")
        X_imp = self.imputer.fit_transform(X_train)

        self.scaler_x = StandardScaler()
        X_scaled = self.scaler_x.fit_transform(X_imp)

        self.scaler_y = StandardScaler()
        y_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

        # 2. Chronological Train / Val Split for Early Stopping
        n_total = len(X_scaled)
        n_sub_train = int(n_total * (1.0 - val_ratio))

        X_sub_train, y_sub_train = X_scaled[:n_sub_train], y_scaled[:n_sub_train]
        X_sub_val, y_sub_val = X_scaled[n_sub_train:], y_scaled[n_sub_train:]

        # 3. Create 3D Sequences
        X_train_seq, y_train_seq = self.create_sequences(X_sub_train, y_sub_train, self.lookback)
        
        # Include lookback padding for validation sequence so it starts cleanly from the boundary
        val_start_idx = max(0, n_sub_train - self.lookback + 1)
        X_val_seq, y_val_seq = self.create_sequences(
            X_scaled[val_start_idx:], y_scaled[val_start_idx:], self.lookback
        )

        train_dataset = TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(y_train_seq))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        X_val_tensor = torch.from_numpy(X_val_seq)
        y_val_tensor = torch.from_numpy(y_val_seq)

        # 4. Initialize Network
        input_size = len(self.feature_names)
        self.model = PyTorchLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            dense_units=self.dense_units,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # 5. Training Loop with Early Stopping
        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0

        self.history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            epoch_train_losses = []
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_losses.append(loss.item())

            avg_train_loss = float(np.mean(epoch_train_losses))
            self.history["train_loss"].append(avg_train_loss)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_tensor)
                val_loss = float(criterion(val_pred, y_val_tensor).item())
                self.history["val_loss"].append(val_loss)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Epoch {epoch:02d}/{self.max_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch} (Best Val Loss: {best_val_loss:.4f})")
                    break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        self.is_fitted = True
        return self

    def predict_sequences(self, X_seq: np.ndarray) -> np.ndarray:
        """Predict given already-constructed 3D tensor sequences."""
        if not self.is_fitted or self.model is None or self.scaler_y is None:
            raise RuntimeError("LSTMForecaster must be fitted before predict() is called.")
        
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X_seq.astype(np.float32))
            norm_preds = self.model(tensor_x).numpy().reshape(-1, 1)

        # Invert target scaling
        orig_preds = self.scaler_y.inverse_transform(norm_preds).flatten()
        return orig_preds

    def predict_test_boundary(
        self,
        full_X: pd.DataFrame,
        test_start_idx: int,
        test_end_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Construct continuous lookback sequences across the train/test boundary to predict test rows.

        Args:
            full_X: Complete feature matrix across train and test partitions.
            test_start_idx: Index in full_X where the test period begins.
            test_end_idx: Optional index in full_X where the test period ends.

        Returns:
            np.ndarray: Predictions for rows from test_start_idx to test_end_idx.
        """
        if not self.is_fitted or self.imputer is None or self.scaler_x is None:
            raise RuntimeError("Model must be fitted before prediction.")

        # Transform full_X using pre-fitted transformers
        X_imp = self.imputer.transform(full_X[self.feature_names])
        X_scaled = self.scaler_x.transform(X_imp)

        # Lookback start index
        lookback_start = test_start_idx - self.lookback + 1
        if lookback_start < 0:
            raise ValueError(f"Not enough history before test_start_idx ({test_start_idx}) for lookback {self.lookback}")

        if test_end_idx is not None:
            X_slice = X_scaled[lookback_start:test_end_idx]
        else:
            X_slice = X_scaled[lookback_start:]

        X_seq, _ = self.create_sequences(X_slice, y=None, lookback=self.lookback)

        return self.predict_sequences(X_seq)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate forecasts for feature DataFrame X by building internal lookback sequences.

        Args:
            X: Feature DataFrame with at least `lookback` rows.

        Returns:
            np.ndarray: Array of predictions corresponding to rows from (lookback - 1) to end.
        """
        if not self.is_fitted or self.imputer is None or self.scaler_x is None:
            raise RuntimeError("Model must be fitted before prediction.")
        X_imp = self.imputer.transform(X[self.feature_names])
        X_scaled = self.scaler_x.transform(X_imp)
        X_seq, _ = self.create_sequences(X_scaled, y=None, lookback=self.lookback)
        return self.predict_sequences(X_seq)

    def save_model(self, file_path: Union[str, Path]) -> None:
        """Save model checkpoint, scaler parameters, and architecture settings."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save an unfitted model.")
        
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "feature_names": self.feature_names,
            "lookback": self.lookback,
            "hidden_size": self.hidden_size,
            "dense_units": self.dense_units,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "imputer": self.imputer,
            "scaler_x": self.scaler_x,
            "scaler_y": self.scaler_y,
            "history": self.history,
        }
        torch.save(checkpoint, str(path))

    def load_model(self, file_path: Union[str, Path]) -> "LSTMForecaster":
        """Load saved checkpoint and restore model and scaler parameters."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {path.resolve()}")

        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        self.feature_names = checkpoint["feature_names"]
        self.lookback = checkpoint["lookback"]
        self.hidden_size = checkpoint["hidden_size"]
        self.dense_units = checkpoint["dense_units"]
        self.num_layers = checkpoint["num_layers"]
        self.dropout = checkpoint["dropout"]

        # Restore imputer and scalers
        self.imputer = checkpoint["imputer"]
        self.scaler_x = checkpoint["scaler_x"]
        self.scaler_y = checkpoint["scaler_y"]

        self.model = PyTorchLSTM(
            input_size=len(self.feature_names),
            hidden_size=self.hidden_size,
            dense_units=self.dense_units,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})
        self.is_fitted = True
        return self
