# Explanation for TT-008 — Dot Product (NumPy)

The dot product of vectors x, y (same length n) is Σ x[i]*y[i] from i=0 to n-1.

## Steps in `solution.py`:

1. Convert inputs: `x = np.array(x)`, `y = np.array(y)` – handles lists/arrays.
2. Validate: `if len(x) != len(y): raise ValueError("Vectors must be of the same length")`.
3. Compute elementwise: `x * y` (broadcast multiply).
4. Sum: `np.sum(...)`.
5. Cast: `np.float64(...)` for scalar float output.

## Notes on behavior
- Fully vectorized O(n) time.
- Works for 1D only (length check on 1D).
- `np.sum(x * y)` more efficient than `np.dot` for this impl.
- Error on mismatch prevents invalid computation.
- Output always float64 scalar.

## Output shape
Always single float64 scalar.

