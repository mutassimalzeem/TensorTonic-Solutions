# Explanation for TT-006 - Adam

Adam is an adaptive optimization algorithm commonly used to train neural networks.

It improves on plain gradient descent by using two running averages:

- `m`: the first moment estimate, similar to momentum.
- `v`: the second moment estimate, similar to RMSProp.

These estimates help the optimizer move smoothly and scale updates based on recent gradient behavior.

## Function signature

```python
adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)
```

## Step-by-step behavior

1. Convert inputs into NumPy arrays with `dtype=float`.
2. Update the momentum term:

```python
m = beta1 * m + (1 - beta1) * grad
```

3. Update the squared-gradient term:

```python
v = beta2 * v + (1 - beta2) * grad ** 2
```

4. Apply bias correction:

```python
m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)
```

5. Update the parameter:

```python
param = param - (m_hat / ((v_hat + eps) ** 0.5)) * lr
```

6. Return the updated values:

```python
return param, m, v
```

## Why bias correction is needed

At the start of training, `m` and `v` are often initialized to zero. This makes the early moving averages biased toward zero.

Adam fixes this by dividing by:

```python
1 - beta1 ** t
1 - beta2 ** t
```

This correction is especially important during the first few update steps.

## Role of each parameter

- `param`: the current model parameter or parameter vector.
- `grad`: the gradient for the current step.
- `m`: running average of gradients.
- `v`: running average of squared gradients.
- `t`: current timestep, starting from 1.
- `lr`: learning rate.
- `beta1`: decay rate for the first moment estimate.
- `beta2`: decay rate for the second moment estimate.
- `eps`: small value added for numerical stability.

## Notes

- The function performs one optimizer step, not a full training loop.
- The updated `m` and `v` should be passed into the next call.
- The implementation works with scalars, lists, and NumPy arrays as long as shapes are compatible.
- The function returns NumPy arrays because the inputs are converted with `np.array(...)`.
