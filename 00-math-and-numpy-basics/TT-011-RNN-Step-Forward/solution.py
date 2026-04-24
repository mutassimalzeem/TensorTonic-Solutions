"""Requirements
Compute: pre_act = x_t @ Wx + h_prev @ Wh + b
Apply: h_t = np.tanh(pre_act)
Return h_t as 1D array of shape (H,)
Do not modify inputs in place
No batching: single time-step vectors only"""

import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    x_t = np.atleast_1d(x_t).astype(float)
    h_prev = np.atleast_1d(h_prev).astype(float)
    
    input_dim = x_t.shape[0]
    hidden_dim = h_prev.shape[0]

    if np.isscalar(Wx) and Wx == 0:
        Wx = np.zeros((input_dim, hidden_dim))
    else:
        Wx = np.array(Wx, dtype=float)
        
    if np.isscalar(Wh) and Wh == 0:
        Wh = np.zeros((hidden_dim, hidden_dim))
    else:
        Wh = np.array(Wh, dtype=float)
        
    if np.isscalar(b) and b == 0:
        b = np.zeros(hidden_dim)
    else:
        b = np.array(b, dtype=float)

    # RNN Forward Step Formula:
    # h_t = tanh(x_t * Wx + h_prev * Wh + b)
    preact = (x_t @ Wx) + (h_prev @ Wh) + b
    h_t = np.tanh(preact)

    return h_t

# Running your test case
output = rnn_step_forward(x_t=[0.0,0.0], h_prev=[1.0,-1.0], Wx=0, Wh=np.identity(2), b=0)
print(f"Output h_t: {output}")
# Input: xt=[1.0,0.0],hprev=[0.0,0.0]

# Weights: Wx=I2×2,Wh=0,b=0

# Output: ht=tanh⁡([1.0,0.0])≈[0.7616,0.0]