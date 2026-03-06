"""Fully Connected Network (FCN) for salary regression using PyTorch."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


class FCNRegressor:
    """Salary predictor based on a fully connected neural network.

    Architecture: Linear → BatchNorm → ReLU → Dropout (× N) → Linear.
    Wraps a PyTorch Module so it exposes the same interface as
    ``SalaryRegressor`` (fit / predict / evaluate / save / load).

    Args:
        hidden_dims: Widths of the hidden layers.
        dropout: Dropout probability after each hidden layer.
        lr: Adam learning rate.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        patience: Early-stopping patience (epochs without val-loss improvement).
        device: Torch device string; auto-detects CUDA when ``None``.
    """

    DEFAULT_WEIGHTS_PATH = (
        Path(__file__).parent.parent / "resources" / "nn_model_weights.pt"
    )
    SALARY_MULTIPLIER = 1000.0

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 150,
        batch_size: int = 64,
        patience: int = 15,
        device: Optional[str] = None,
    ) -> None:
        self.hidden_dims = hidden_dims or [256, 256, 128]
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._net: Optional[nn.Module] = None
        self._input_dim: Optional[int] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FCNRegressor":
        """Train the network on the given data.

        Args:
            X: Feature matrix (n_samples, n_features), float32.
            y: Target vector (n_samples,) — salary in thousands of rubles.

        Returns:
            Self for method chaining.
        """
        self._input_dim = X.shape[1]
        self._net = self._build_net(self._input_dim).to(self.device)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_loss = float("inf")
        no_improve = 0
        best_state: Optional[dict] = None

        for epoch in range(1, self.epochs + 1):
            self._net.train()
            epoch_loss = 0.0
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self._net(X_b).squeeze(1), y_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(y_b)

            epoch_loss /= len(dataset)
            scheduler.step(epoch_loss)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict salaries in rubles.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Predicted salaries in rubles.
        """
        if not self._is_fitted or self._net is None:
            raise RuntimeError("Model is not fitted")

        self._net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            preds = self._net(X_t).squeeze(1).cpu().numpy()
        return preds * self.SALARY_MULTIPLIER

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance.

        Args:
            X: Feature matrix.
            y: True targets in thousands of rubles.

        Returns:
            Dict with mae, mse, rmse, r2 metrics.
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        if not self._is_fitted or self._net is None:
            raise RuntimeError("Model is not fitted")

        self._net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_pred = self._net(X_t).squeeze(1).cpu().numpy()

        return {
            "mae": mean_absolute_error(y, y_pred) * self.SALARY_MULTIPLIER,
            "mse": mean_squared_error(y, y_pred) * (self.SALARY_MULTIPLIER ** 2),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))) * self.SALARY_MULTIPLIER,
            "r2": float(r2_score(y, y_pred)),
        }

    def save(self, path: Optional[Path] = None) -> None:
        """Save model weights to file.

        Args:
            path: Save path (default: resources/nn_model_weights.pt).
        """
        if not self._is_fitted or self._net is None:
            raise RuntimeError("Model is not fitted")

        save_path = path or self.DEFAULT_WEIGHTS_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "input_dim": self._input_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
            },
            save_path,
        )

    def load(self, path: Optional[Path] = None) -> "FCNRegressor":
        """Load model weights from file.

        Args:
            path: Load path (default: resources/nn_model_weights.pt).

        Returns:
            Self for method chaining.
        """
        load_path = path or self.DEFAULT_WEIGHTS_PATH
        if not load_path.exists():
            raise FileNotFoundError(f"Model weights not found at {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)
        self._input_dim = checkpoint["input_dim"]
        self.hidden_dims = checkpoint["hidden_dims"]
        self.dropout = checkpoint["dropout"]
        self._net = self._build_net(self._input_dim).to(self.device)
        self._net.load_state_dict(checkpoint["state_dict"])
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_net(self, input_dim: int) -> nn.Module:
        """Construct the FCN architecture."""
        layers: List[nn.Module] = []
        prev = input_dim
        for dim in self.hidden_dims:
            layers += [
                nn.Linear(prev, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
            ]
            prev = dim
        layers.append(nn.Linear(prev, 1))

        net = nn.Sequential(*layers)
        for module in net.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        return net
