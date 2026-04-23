import numpy as np

def linear_regression_closed_form(X, y):
    np.array(X)
    np.array(y)

    w1 = np.linalg.inv(np.dot(np.transpose(X), X))
    w2 = np.dot(np.transpose(X), y)
    w = np.dot(w1, w2)

    return w