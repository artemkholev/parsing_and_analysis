"""Training script for the FCN salary prediction model with MLflow tracking.

Usage (from project root):
    python scripts/train_nn.py path/to/x_data.npy path/to/y_data.npy

Workflow:
    1. Load preprocessed arrays produced by ``pipeline_app.py``.
    2. Split into train/test (70/30).
    3. Train a Fully Connected Network (FCN).
    4. Log parameters, metrics and the model artifact to MLflow.
    5. Save trained weights to ``resources/nn_model_weights.pt``.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
from sklearn.model_selection import train_test_split

from model import FCNRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------------------------
# MLflow configuration
# ------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://kamnsv.com:55000/"
EXPERIMENT_NAME = "LIne Regression HH"
RUN_NAME = "kholev_artem_fcn"

# ------------------------------------------------------------------
# Hyper-parameters
# ------------------------------------------------------------------
HIDDEN_DIMS = [256, 256, 128]
DROPOUT = 0.1
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 150
PATIENCE = 15
TEST_SIZE = 0.3
RANDOM_STATE = 42


def load_data(x_path: Path, y_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load feature matrix and target vector from .npy files.

    Args:
        x_path: Path to the feature array.
        y_path: Path to the target array.

    Returns:
        Tuple of (X, y) numpy arrays.
    """
    return np.load(x_path).astype("float32"), np.load(y_path).astype("float32")


def log_to_mlflow(
    model: FCNRegressor,
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> str:
    """Log parameters, metrics and model artifact to MLflow.

    Args:
        model: Fitted FCNRegressor instance.
        train_metrics: Metrics evaluated on the training set.
        test_metrics: Metrics evaluated on the test set.

    Returns:
        MLflow run ID.
    """
    with mlflow.start_run(run_name=RUN_NAME) as run:
        run_id = run.info.run_id
        client = mlflow.MlflowClient()

        client.log_param(run_id, "hidden_dims", str(HIDDEN_DIMS))
        client.log_param(run_id, "dropout", DROPOUT)
        client.log_param(run_id, "lr", LEARNING_RATE)
        client.log_param(run_id, "batch_size", BATCH_SIZE)
        client.log_param(run_id, "epochs", EPOCHS)
        client.log_param(run_id, "patience", PATIENCE)
        client.log_param(run_id, "test_size", TEST_SIZE)
        client.log_param(run_id, "optimizer", "Adam")
        client.log_param(run_id, "scheduler", "ReduceLROnPlateau")

        client.log_metric(run_id, "train_mae", train_metrics["mae"])
        client.log_metric(run_id, "train_rmse", train_metrics["rmse"])
        client.log_metric(run_id, "train_r2", train_metrics["r2"])
        client.log_metric(run_id, "test_mae", test_metrics["mae"])
        client.log_metric(run_id, "test_rmse", test_metrics["rmse"])
        client.log_metric(run_id, "r2_score_test", test_metrics["r2"])

        try:
            mlflow.pytorch.log_model(
                pytorch_model=model.net,
                name=RUN_NAME,
                registered_model_name=RUN_NAME,
            )
        except mlflow.exceptions.MlflowException as exc:
            print(f"Warning: model artifact upload failed: {exc}", file=sys.stderr)

    return run_id


def main() -> int:
    """Entry point for FCN training with MLflow.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    if len(sys.argv) != 3:
        print(
            f"Usage: python {sys.argv[0]} path/to/x_data.npy path/to/y_data.npy",
            file=sys.stderr,
        )
        return 1

    x_path = Path(sys.argv[1])
    y_path = Path(sys.argv[2])

    for path in (x_path, y_path):
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            return 1

    print("Loading data...")
    x_data, y_data = load_data(x_path, y_path)
    print(f"  X shape: {x_data.shape}")
    print(f"  y shape: {y_data.shape}")

    print(
        f"\nSplitting data ({int((1 - TEST_SIZE) * 100)}% train / "
        f"{int(TEST_SIZE * 100)}% test)..."
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(x_train)}  Test: {len(x_test)}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\nMLflow tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"Experiment          : {EXPERIMENT_NAME}")
    print(f"Run name            : {RUN_NAME}")

    print("\nTraining FCN model...")
    model = FCNRegressor(
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
    )
    model.fit(x_train, y_train)

    print("\nEvaluating on train set:")
    train_metrics = model.evaluate(x_train, y_train)
    print(f"  MAE:  {train_metrics['mae']:.2f} RUB")
    print(f"  RMSE: {train_metrics['rmse']:.2f} RUB")
    print(f"  R\u00b2:   {train_metrics['r2']:.4f}")

    print("\nEvaluating on test set:")
    test_metrics = model.evaluate(x_test, y_test)
    print(f"  MAE:  {test_metrics['mae']:.2f} RUB")
    print(f"  RMSE: {test_metrics['rmse']:.2f} RUB")
    print(f"  R\u00b2:   {test_metrics['r2']:.4f}")

    run_id = log_to_mlflow(model, train_metrics, test_metrics)
    print(f"\nRUN_ID: {run_id}")
    print(f"r2_score_test logged to MLflow: {test_metrics['r2']:.4f}")

    model.save()
    print(f"Model weights saved to: {model.DEFAULT_WEIGHTS_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
