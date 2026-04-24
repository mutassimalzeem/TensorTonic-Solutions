import numpy as np

def euclidean_distance(x, y):

    x = np.array(x)
    y = np.array(y)

    if x.shape == y.shape:
        z = np.sum((x - y)**2)
        return np.sqrt(z)
    else:
        raise ValueError("Incompatible shapes for Euclidean distance")