# Notes for TT-003 — Softmax

- The function converts the input to a NumPy array before performing any computations.
- It uses a numerically stable softmax formulation by subtracting the maximum value along the last axis before exponentiation.
- `axis=-1` and `keepdims=True` are used so the implementation works for both 1D and multi-dimensional arrays.
- The output preserves the input shape and represents probabilities that sum to 1 along the last axis.
- The implementation is fully vectorized and does not use Python loops.
- For a scalar input, the output is a single-element NumPy array with value `1.0`.
