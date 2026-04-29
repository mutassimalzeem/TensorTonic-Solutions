import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    # Your code here
    hidden1 = np.dot(x, W1) + b1
    relu = np.maximum(0, hidden1)
    hidden2 = np.dot(relu, W2) + b2
    return hidden2