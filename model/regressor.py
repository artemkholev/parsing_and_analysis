"""Salary prediction model using Ridge Regression"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class SalaryRegressor:
    """Ridge Regression model for salary prediction with L2 regularization"""

    DEFAULT_WEIGHTS_PATH = Path(__file__).parent.parent / "resources" / "model_weights.joblib"
    SALARY_MULTIPLIER = 1000.0

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: L2 regularization strength
        """
        self._model = Ridge(alpha=alpha)
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SalaryRegressor":
        """
        Train model on given data

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)

        Returns:
            Self for method chaining
        """
        self._model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict salaries for given features

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted salaries in rubles
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted")

        predictions = self._model.predict(X)
        return predictions * self.SALARY_MULTIPLIER

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate model performance

        Args:
            X: Feature matrix
            y: True targets (in thousands of rubles)

        Returns:
            Dict with mae, mse, rmse, r2 metrics
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted")

        y_pred = self._model.predict(X)

        return {
            "mae": mean_absolute_error(y, y_pred) * self.SALARY_MULTIPLIER,
            "mse": mean_squared_error(y, y_pred) * (self.SALARY_MULTIPLIER**2),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)) * self.SALARY_MULTIPLIER,
            "r2": r2_score(y, y_pred),
        }

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save model weights to file

        Args:
            path: Save path (default: resources/model_weights.joblib)
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted")

        save_path = path or self.DEFAULT_WEIGHTS_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {"model": self._model, "is_fitted": self._is_fitted},
            save_path,
        )

    def load(self, path: Optional[Path] = None) -> "SalaryRegressor":
        """
        Load model weights from file

        Args:
            path: Load path (default: resources/model_weights.joblib)

        Returns:
            Self for method chaining
        """
        load_path = path or self.DEFAULT_WEIGHTS_PATH

        if not load_path.exists():
            raise FileNotFoundError(f"Model weights not found at {load_path}")

        data = joblib.load(load_path)
        self._model = data["model"]
        self._is_fitted = data["is_fitted"]

        return self

    @property
    def coefficients(self) -> np.ndarray:
        """Model coefficients"""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted")
        return self._model.coef_

    @property
    def intercept(self) -> float:
        """Model intercept"""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted")
        return self._model.intercept_
