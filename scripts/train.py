"""Training script for salary prediction model"""

import sys
import warnings
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=RuntimeWarning)

from model import SalaryRegressor


def load_data(x_path: Path, y_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load training data from numpy files

    Args:
        x_path: Path to features file
        y_path: Path to targets file

    Returns:
        Tuple of (X, y) arrays
    """
    X = np.load(x_path)
    y = np.load(y_path)
    return X, y


def main() -> int:
    """
    Main training function

    Returns:
        Exit code (0 - success, 1 - error)
    """
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} path/to/x_data.npy path/to/y_data.npy")
        return 1

    x_path = Path(sys.argv[1])
    y_path = Path(sys.argv[2])

    if not x_path.exists():
        print(f"Error: X data file not found: {x_path}")
        return 1

    if not y_path.exists():
        print(f"Error: y data file not found: {y_path}")
        return 1

    print("Loading data...")
    X, y = load_data(x_path, y_path)
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    print("\nSplitting data (70% train, 30% test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"  Train size: {len(X_train)}")
    print(f"  Test size: {len(X_test)}")

    print("\nTraining Ridge Regression model...")
    model = SalaryRegressor(alpha=1.0)
    model.fit(X_train, y_train)

    print("\nEvaluating on train set:")
    train_metrics = model.evaluate(X_train, y_train)
    print(f"  MAE:  {train_metrics['mae']:.2f} RUB")
    print(f"  RMSE: {train_metrics['rmse']:.2f} RUB")
    print(f"  R2:   {train_metrics['r2']:.4f}")

    print("\nEvaluating on test set:")
    test_metrics = model.evaluate(X_test, y_test)
    print(f"  MAE:  {test_metrics['mae']:.2f} RUB")
    print(f"  RMSE: {test_metrics['rmse']:.2f} RUB")
    print(f"  R2:   {test_metrics['r2']:.4f}")

    print("\nSaving model weights...")
    model.save()
    print(f"  Saved to: {model.DEFAULT_WEIGHTS_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
