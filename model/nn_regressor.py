from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class FCNRegressor:
    """Salary predictor based on a fully connected neural network.

    Architecture: Linear -> BatchNorm -> ReLU -> Dropout (x N) -> Linear.
    Exposes fit / predict / evaluate / save / load interface.

    Args:
        hidden_dims: Widths of the hidden layers.
        dropout: Dropout probability after each hidden layer.
        lr: Adam learning rate.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        patience: Early-stopping patience (epochs without improvement).
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

    @property
    def net(self) -> nn.Module:
        """Underlying PyTorch module (must be fitted first)."""
        if self._net is None:
            raise RuntimeError("Model is not fitted")
        return self._net

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "FCNRegressor":
        """Train the network on the given data.

        Args:
            x_train: Feature matrix (n_samples, n_features), float32.
            y_train: Target vector (n_samples,) in thousands of rubles.

        Returns:
            Self for method chaining.
        """
        self._input_dim = x_train.shape[1]
        self._net = self._build_net(self._input_dim).to(self.device)

        x_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        best_state = self._run_training_loop(loader, len(dataset))
        if best_state is not None:
            self._net.load_state_dict(best_state)

        self._is_fitted = True
        return self

    def _run_training_loop(
        self,
        loader: torch.utils.data.DataLoader,
        n_samples: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Execute the training loop with early stopping.

        Args:
            loader: DataLoader for training data.
            n_samples: Total number of training samples.

        Returns:
            State dict of the best model, or ``None`` if no step improved loss.
        """
        assert self._net is not None
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_loss = float("inf")
        no_improve = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None

        for _ in range(1, self.epochs + 1):
            self._net.train()
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self._net(x_batch).squeeze(1), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(y_batch)

            epoch_loss /= n_samples
            scheduler.step(epoch_loss)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {
                    k: v.cpu().clone() for k, v in self._net.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        return best_state

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        """Predict salaries in rubles.

        Args:
            x_data: Feature matrix (n_samples, n_features).

        Returns:
            Predicted salaries in rubles.
        """
        if not self._is_fitted or self._net is None:
            raise RuntimeError("Model is not fitted")

        self._net.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_data, dtype=torch.float32).to(self.device)
            preds = self._net(x_tensor).squeeze(1).cpu().numpy()
        return preds * self.SALARY_MULTIPLIER

    def evaluate(self, x_data: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            x_data: Feature matrix.
            y_true: True targets in thousands of rubles.

        Returns:
            Dict with mae, mse, rmse, r2 metrics.
        """
        if not self._is_fitted or self._net is None:
            raise RuntimeError("Model is not fitted")

        self._net.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_data, dtype=torch.float32).to(self.device)
            y_pred = self._net(x_tensor).squeeze(1).cpu().numpy()

        return {
            "mae": mean_absolute_error(y_true, y_pred) * self.SALARY_MULTIPLIER,
            "mse": mean_squared_error(y_true, y_pred) * (self.SALARY_MULTIPLIER**2),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))
            * self.SALARY_MULTIPLIER,
            "r2": float(r2_score(y_true, y_pred)),
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
