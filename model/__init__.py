"""Model module for salary prediction."""

from model.nn_regressor import FCNRegressor
from model.regressor import SalaryRegressor

__all__ = ["SalaryRegressor", "FCNRegressor"]
