import numpy as np

"""Requirements
Must work for lists or NumPy arrays
Must return a float
Must be vectorized (no Python element loops)
Must handle 1D arrays only
Must raise ValueError for mismatched lengths"""

def dot_product(x, y):
    x = np.array(x)
    y = np.array(y)

    if len(x) != len(y):
        raise ValueError("Vectors must be of the same length")
    return np.float64(np.sum(x * y))

print(dot_product([1, 2, 3], [4, 5, 6]))