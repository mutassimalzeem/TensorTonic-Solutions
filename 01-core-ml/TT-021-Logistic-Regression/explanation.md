# Explanation for TT-021 - Logistic Regression

Logistic regression is used for binary classification. It learns a linear score for each sample and passes that score through a sigmoid function so the output can be interpreted as a probability.

## Formula

For input matrix `X`, weight vector `w`, and bias `b`:

```python
z = X @ w + b
p = 1 / (1 + exp(-z))
```

The binary cross entropy loss for a batch is:

```python
loss = -mean(y * log(p) + (1 - y) * log(1 - p))
```

## Gradient updates

The implementation uses batch gradient descent. For `n` samples:

```python
grad_w = (1 / n) * X.T @ (p - y)
grad_b = (1 / n) * sum(p - y)
```

Then the parameters are updated:

```python
w = w - lr * grad_w
b = b - lr * grad_b
```

## Implementation details

- `np.array(X, dtype=float)` converts feature data into a numeric array.
- `np.array(y, dtype=float)` converts labels into numeric values.
- `_sigmoid(z)` uses a numerically stable form to reduce overflow risk for very large positive or negative logits.
- `np.clip(p, 1e-15, 1 - 1e-15)` prevents `log(0)` during loss calculation.
- The gradients are vectorized across the full batch.
- The function returns the learned weight vector and bias as `(w, b)`.

## Output shape

Input `X` with shape `(n_samples, n_features)` returns:

```python
w.shape == (n_features,)
b is a scalar float
```
