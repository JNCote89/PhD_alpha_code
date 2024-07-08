from sklearn.metrics import root_mean_squared_error
import numpy as np


def rmse(y_true: np.array, y_pred: np.array, precision: float = 3):
    return round(root_mean_squared_error(y_true, y_pred), precision)
