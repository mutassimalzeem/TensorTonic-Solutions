import numpy as np

def softmax(x):
    x = np.array(x)

    
    shifted = x - np.max(x, axis = -1, keepdims= True)
    exp_vals = np.exp(shifted)
    denominator = np.sum(exp_vals, axis = -1, keepdims= True)
    output = exp_vals / denominator

    return output