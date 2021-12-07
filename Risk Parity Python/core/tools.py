import quadprog

import numpy as np


def to_column_matrix(x):
    if x.shape[1] != 1:
        x = np.transpose(x)
    if x.shape[1] == 1:
        return x
    else:
        raise ValueError("x is not a vector")


def to_array(x):
    if x is None:
        return None
    elif (len(x.shape)) == 1:
        return x
    if x.shape[1] != 1:
        x = x.T
    return np.squeeze(np.asarray(x))


