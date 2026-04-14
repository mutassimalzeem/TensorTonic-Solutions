# Explanation for TT-001 — Sigmoid (NumPy)

The sigmoid function is:

```
sigmoid(x) = 1 / (1 + exp(-x))
```

The implementation in `solution.py` uses NumPy to apply this formula elementwise.

Steps performed by the code:

1. Convert the input to a NumPy array using `np.array(x)`.
2. Compute `np.exp(-x)` for every element.
3. Evaluate `1 / (1 + np.exp(-x))` to obtain the sigmoid output.

Because NumPy operations are vectorized, the function works for scalars, Python lists, and arrays of any shape.

## Notes on behavior

- The output values are in the range `(0, 1)`.
- The return value preserves the input shape.
- For large negative inputs, the direct `np.exp(-x)` expression may trigger a runtime warning from NumPy, but the formula is the standard sigmoid expression.

## Output shape

The function returns a NumPy array with the same shape as the original input.
