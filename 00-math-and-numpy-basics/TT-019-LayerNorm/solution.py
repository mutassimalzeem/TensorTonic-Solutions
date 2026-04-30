import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    # Your code here
    mean_x = np.mean(x, axis= -1, keepdims = True)

    var = np.var(x, axis = -1, keepdims = True)
    #var = np.sum((x - mean_x)**2) / len(x)
    norm = (x - mean_x) / np.sqrt(var + eps)
    output = gamma * norm + beta

    return output

print(layer_norm(x = [[1.0, 2.0, 3.0, 4.0]], gamma = [1.0, 1.0, 1.0, 1.0], beta = [0.0, 0.0, 0.0, 0.0]))
print(layer_norm(x = [[1.0, 2.0, 3.0, 4.0]], gamma = [2.0, 2.0, 2.0, 2.0], beta = [1.0, 1.0, 1.0, 1.0]))
print(layer_norm(x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], gamma = [1.0, 1.0, 1.0], beta = [0.0, 0.0, 0.0]))