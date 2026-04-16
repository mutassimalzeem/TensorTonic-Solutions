# Explanation for TT-003 — Softmax

The softmax function converts a vector of scores into probabilities that sum to 1 along the last axis.

The implementation in `solution.py` performs these steps:

1. Convert the input to a NumPy array using `np.array(x)`.
2. Compute a shifted version of the input by subtracting the maximum value along the last axis: `x - np.max(x, axis=-1, keepdims=True)`.
3. Compute `np.exp(shifted)` elementwise.
4. Sum the exponentials along the last axis with `np.sum(exp_vals, axis=-1, keepdims=True)`.
5. Divide the exponentials by the sum to obtain normalized probabilities.

## Behavior

- The function returns an array of the same shape as the input.
- Values are normalized along the last axis, so each slice along that axis sums to 1.
- The subtraction of the maximum value improves numerical stability and avoids large exponent values.
- The implementation works for scalars, 1D arrays, and multi-dimensional arrays.

## Notes

- For scalar inputs, the softmax result becomes `array([1.0])` after normalization.
- The use of `keepdims=True` preserves dimensions during reduction, allowing the division to broadcast correctly.
- Since the function is vectorized, it avoids explicit Python loops and works efficiently with NumPy inputs.
