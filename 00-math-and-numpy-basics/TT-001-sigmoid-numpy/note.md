# Notes for TT-001 — Sigmoid (NumPy)

- The sigmoid function maps real values to values between 0 and 1.
- In `solution.py`, the implementation uses `np.array(x)` to convert scalars, Python lists, and existing NumPy arrays into a NumPy array.
- The function then computes the sigmoid output with the formula `1 / (1 + np.exp(-x))`.
- This makes the implementation vectorized and able to handle inputs of any shape.
- The return value preserves the input shape.
- For very large negative values, the direct `np.exp(-x)` computation may emit a runtime warning due to overflow, but the formula is correct for typical inputs.
- Useful test inputs include scalars, lists, 1D arrays, and 2D arrays.
