import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # Initializations
    w = np.zeros(np.shape(X)[1],)
    b = 0.0

    for i in range(steps):

        #   Linear Combination
        z = np.dot(X, w) + b

        #   Sigmoid
        p = _sigmoid(z)
        p = np.clip(p, 1e-15, 1 - 1e-15)    #   For numerical stability


        #   Binary Cross Entropy Loss Calculation
        n = len(X)
        
        L = -(1/n) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        #   Use gradient descent to minimize the loss function
        grad_l_w = (1 / n) * (np.transpose(X) @ (p - y))
        grad_l_b = (1 / n) * np.sum(p - y)

        # grad_l_w = (p - y) * X
        # grad_l_b = (p - y)

        #   Weights and Biases
        w = w - lr * grad_l_w
        b = b - lr * grad_l_b

    return (w, b)
