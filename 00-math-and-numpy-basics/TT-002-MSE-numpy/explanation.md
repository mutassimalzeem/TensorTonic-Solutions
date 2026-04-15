# Explanation for TT-002 — Mean Squared Error (NumPy)

The mean squared error (MSE) measures the average squared difference between predicted values and actual target values.

The implementation in `solution.py` follows these steps:

1. Convert both inputs to NumPy arrays with `np.array(y_pred)` and `np.array(y_true)`.
2. Compare their shapes; if shapes differ, return `None`.
3. Compute the squared difference elementwise: `(y_pred - y_true) ** 2`.
4. Return the mean of the squared differences with `np.mean(...)`.

## Behavior

- If `y_pred` and `y_true` have the same shape, the function returns a float representing the MSE.
- If the shapes do not match, the function returns `None`.
- The implementation works with scalars, Python lists, and NumPy arrays.

## Formula

```
MSE = mean((y_pred - y_true)**2)
```

## Notes

- The output is a scalar float.
- The use of NumPy ensures the computation is vectorized and efficient.
