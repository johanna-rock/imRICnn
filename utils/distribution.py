import numpy as np


def loguniform(lower_limit, upper_limit, size):
    assert (lower_limit != 0 and upper_limit != 0)
    return np.exp(np.random.uniform(np.log(lower_limit), np.log(upper_limit), size))
