import numpy as np


def confidence_bounds(mean: float, var: float) -> tuple[int, int]:
    lower_bound = mean - 1.96 * np.sqrt(var)
    upper_bound = mean + 1.96 * np.sqrt(var)

    return lower_bound, upper_bound
