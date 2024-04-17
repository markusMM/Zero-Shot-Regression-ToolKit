import numpy as np


def fill_na(x: np.ndarray, rep=0):
    x[x != x] = rep  # noqa
    return x
