import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    x = np.array(x, dtype=float)
    gamma = np.array(gamma, dtype=float)
    beta = np.array(beta, dtype=float)

    if x.ndim == 4:
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)

        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)
    else:
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)

    x_norm = (x - mean) / np.sqrt(var + eps)
    y = gamma * x_norm + beta

    return y
