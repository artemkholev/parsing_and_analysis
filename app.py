"""Salary prediction CLI - outputs predicted salaries in rubles"""

import signal
import sys
import warnings
from pathlib import Path

import numpy as np

from model import SalaryRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)
signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def main() -> int:
    """
    Main prediction function

    Returns:
        Exit code (0 - success, 1 - error)
    """
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} path/to/x_data.npy", file=sys.stderr)
        return 1

    x_path = Path(sys.argv[1])

    if not x_path.exists():
        print(f"Error: File not found: {x_path}", file=sys.stderr)
        return 1

    if x_path.suffix.lower() != ".npy":
        print(f"Error: Expected .npy file, got: {x_path.suffix}", file=sys.stderr)
        return 1

    try:
        X = np.load(x_path)
    except Exception as e:
        print(f"Error loading file: {e}", file=sys.stderr)
        return 1

    model = SalaryRegressor()

    try:
        model.load()
    except FileNotFoundError:
        print("Error: Model weights not found. Run train.py first.", file=sys.stderr)
        return 1

    predictions = model.predict(X)

    for salary in predictions:
        print(f"{salary:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
