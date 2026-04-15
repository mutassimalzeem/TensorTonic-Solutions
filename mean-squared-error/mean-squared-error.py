import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    if np.shape(y_pred) == np.shape(y_true):
        return np.mean(np.square(y_pred - y_true))
        
    else:
        return None