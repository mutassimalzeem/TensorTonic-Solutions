
import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    param = np.array(param, dtype=float)
    grad = np.array(grad, dtype=float)
    m = np.array(m, dtype=float)
    v = np.array(v, dtype=float)


    m = beta1*m + (1 - beta1) * grad            #Momentum
    v = beta2 * v + (1 - beta2) * grad ** 2     #  RMSProp
    
    m_hat = m / (1 - beta1 ** t)

    v_hat = v / (1 - (beta2 ** t))

    param = param - (m_hat / ((v_hat + eps) ** 0.5)) * lr

    return param, m, v

print(adam_step(param=[1.0, 2.0], m=[0, 0], v=[0, 0], grad=[0, 0], t=1, lr=0.001))