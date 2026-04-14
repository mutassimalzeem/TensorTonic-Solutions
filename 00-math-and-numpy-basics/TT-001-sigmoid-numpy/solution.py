import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x = np.array(x)     #   makes it vectorized, so it works for scalars, lists, and NumPy arrays.
    
    z = 1 / (1 + np.exp(-x))

    return z